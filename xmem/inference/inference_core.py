import cv2
import numpy as np
import torch

from xmem.inference.track import Track
from xmem.inference.memory_manager import MemoryManager
from xmem.model.network import XMem as XMemModel
from xmem.model.aggregate import aggregate
from xmem.util.tensor_util import pad_divide_by, unpad
from .config import DEFAULT_CONFIG
from ..checkpoint import ensure_checkpoint

from scipy.optimize import linear_sum_assignment

device = 'cuda'

class XMem(torch.nn.Module):
    def __init__(self, config, checkpoint_path=None, map_location=None, Track=Track):
        super().__init__()
        self.config = config = {**DEFAULT_CONFIG, **(config or {})}
        checkpoint_path = checkpoint_path or ensure_checkpoint()
        self.map_location = map_location
        self.network = XMemModel(config, checkpoint_path, map_location)
        self.network.eval()

        # box tracks
        self.Track = Track
        self.tracks = {}

        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)

        # a kernel to make tiny segmentatation maps a bit bigger (e.g. toothpicks)
        self.dilate_size_threshold = config.get('dilate_size_threshold')
        kernel_size = config.get('dilation_kernel_size') or 0
        self.kernel = np.ones((kernel_size, kernel_size), dtype=np.int32)

        self.tentative_frame_count = config.get('tentative_frames') or 0
        self.track_max_age = config.get('max_age') or None

        self.clear_memory()
        self.next_label = 0

    def clear_memory(self):
        self.curr_ti = 0
        self.last_mem_ti = 0
        self.track_ids = []
        self.tracks.clear()
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        self.memory = MemoryManager(config=self.config)

    def delete_object_id(self, object_id):
        index = self.track_ids.index(object_id)
        self.memory.delete_object_id(object_id, index)
        del self.track_ids[index]

    def update_config(self, config):
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)
        self.memory.update_config(config)

    def _set_track_ids(self, labels):
        if isinstance(labels, int):
            labels = labels - len(self.track_ids)
            if labels < 0: 
                raise RuntimeError("You cant reduce label count like this.")
            elif labels == 0:
                return
            new = list(range(self.next_label, self.next_label + labels))
            print('new', new, labels, len(self.track_ids))
            labels = self.track_ids + new
        self.track_ids = labels
        self.next_label = max(max(self.track_ids) + 1, self.next_label)
        print("updated labels", labels)

    def forward(self, image, mask=None, valid_track_ids=None, no_update=False, binarize_mask=True):
        # preprocess image

        # image: 3*H*W
        # mask: num_objects*H*W or None
        image, pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0) # add the batch dimension

        # exit early, there is nothing to track
        engaged = self.memory.work_mem.engaged()
        if (mask is None or not mask.shape[0]) and not engaged:
            input_track_ids = None if mask is None else np.array([])
            return (
                torch.ones((0 if binarize_mask else 1, *image.shape[-2:]), device=image.device), 
                np.array([]), 
                input_track_ids)

        # should we update memory?
        is_mem_frame = (
            (self.curr_ti-self.last_mem_ti >= self.mem_every) or 
            (mask is not None)
        ) and (not no_update)

        # do we need to compute segmentation masks?
        need_segment = engaged and (
            valid_track_ids is None or 
            len(self.track_ids) != len(valid_track_ids)
        )

        # should we do a deep memory update?
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame) or  # synchronized
            (not self.deep_update_sync and self.curr_ti-self.last_deep_update_ti >= self.deep_update_every) # no-sync
        ) and (not no_update)

        # is this a normal memory update?
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (not no_update)

        # encode image

        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(
            image, 
            need_ek=(self.enable_long_term or need_segment), 
            need_sk=is_mem_frame)
        # f16: torch.Size([1, 1024, H/16, W/16])
        # f8:  torch.Size([1, 512,  H/8,  W/8])
        # f4:  torch.Size([1, 256,  H/4,  W/4])
        multi_scale_features = (f16, f8, f4)

        # predict segmentation mask for current timestamp

        pred_prob_no_bg = pred_prob_with_bg = None
        if need_segment:
            # shape [1, ?]
            memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)
            # get track masks
            hidden, _, pred_prob_with_bg = self.network.segment(
                multi_scale_features, memory_readout, 
                self.memory.get_hidden(), 
                h_out=is_normal_update, 
                strip_bg=False)
            # pred_prob_with_bg: [ batch, class_id, H, W ] in [0, 1]
            # remove batch dim
            pred_prob_with_bg = pred_prob_with_bg[0]
            pred_prob_no_bg = pred_prob_with_bg[1:]
            if is_normal_update:
                self.memory.set_hidden(hidden)

        # handle external segmentation mask

        input_track_ids = valid_track_ids
        if mask is not None:

            # match object detections with minimum IoU

            # increase the size of very small masks to make tracking easier
            if self.dilate_size_threshold:
                mask = dilate_masks(mask, self.kernel, self.dilate_size_threshold)

            if valid_track_ids is None:
                if pred_prob_no_bg is None:
                    assert not engaged
                    self._set_track_ids(len(mask))
                    input_track_ids = self.track_ids
                else:
                    pred_mask_no_bg = mask_pred_to_binary(pred_prob_with_bg)[1:]
                    mask, input_track_ids, unmatched_rows, new_rows = assign_masks(
                        pred_mask_no_bg, mask, pred_prob_no_bg)
                    if len(new_rows):
                        # print("unmatched/new rows", unmatched_rows, new_rows, len(mask))
                        self._set_track_ids(len(mask))
                    input_track_ids = [self.track_ids[i] for i in input_track_ids]

            # convert input mask

            mask, _ = pad_divide_by(mask, 16)

            if pred_prob_no_bg is not None:
                # convert all masks overlapping with ground truth mask to zero
                pred_prob_no_bg[:, (mask.sum(0) > 0.5)] = 0
                # shift by 1 because mask/pred_prob_no_bg do not contain background
                mask = mask.type_as(pred_prob_no_bg)
                if valid_track_ids is not None:
                    shift_by_one_non_labels = [
                        i for i in range(pred_prob_no_bg.shape[0]) 
                        if (i+1) not in valid_track_ids
                    ]
                    # non-labelled objects are copied from the predicted mask
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]

            # add background back in and convert using softmax
            pred_prob_with_bg = aggregate(mask, dim=0)

            # also create new hidden states
            self.memory.create_hidden_state(len(self.track_ids), key)

        # save to memory

        if is_mem_frame:
            # hidden: torch.Size([1, 3, 64, 2, 2])
            # image segmentation to memory value
            value, hidden = self.network.encode_value(
                image, f16, self.memory.get_hidden(), 
                pred_prob_with_bg[1:].unsqueeze(0), 
                is_deep_update=is_deep_update)
            # save value in memory
            self.memory.add_memory(
                key, shrinkage, value, self.track_ids, 
                selection=selection if self.enable_long_term else None)
            self.last_mem_ti = self.curr_ti

            if is_deep_update:
                self.memory.set_hidden(hidden)
                self.last_deep_update_ti = self.curr_ti

        # get outputs
        pred_mask = unpad(pred_prob_with_bg, pad)
        track_ids = np.asarray(self.track_ids)
        input_track_ids = np.asarray(input_track_ids) if input_track_ids is not None else None

        # handle track deletions
        if input_track_ids is not None:
            _, deleted = self.Track.update_tracks(
                self.tracks, input_track_ids, self.curr_ti,
                n_init=self.tentative_frame_count,
                max_age=self.track_max_age)
            deleted and print('deleted', deleted)
            keep = np.array([True]+[i not in deleted for i in track_ids])
            pred_mask = pred_mask[keep]
            track_ids = track_ids[keep[1:]]
            for i in deleted:
                self.delete_object_id(i)

        # convert probabilities to binary masks using argmax
        if binarize_mask:
            # covert to binary mask
            if pred_mask.ndim == 4:
                pred_mask = pred_mask[0]
            pred_mask_int = torch.argmax(pred_mask, dim=0) - 1
            pred_ids = torch.unique(pred_mask_int)
            pred_ids = pred_ids[pred_ids >= 0]
            pred_mask = pred_ids[:, None, None] == pred_mask_int[None]

            # what should the output format be?
            pred_ids = pred_ids.cpu().numpy()
            track_ids = track_ids[pred_ids]

        self.curr_ti += 1
        # pred_prob_with_bg: [1 + num_objects, H, W]
        # pred_mask: [num_objects, H, W] if binarize_mask else [1 + num_objects, H, W]
        return pred_mask, track_ids, input_track_ids


    def match_iou(self, xmem_mask, other_mask, min_iou=0.4):
        track_ids, other_ids = self.iou_assignment(xmem_mask, other_mask, min_iou)
        track_ids = np.array(self.track_ids, dtype=int)[track_ids]
        return track_ids, other_ids
    
    @staticmethod
    def iou_assignment(first_mask, other_mask, min_iou=0.4):
        iou = mask_iou(first_mask, other_mask)
        iou = iou.cpu().numpy() if isinstance(iou, torch.Tensor) else iou
        track_ids, other_ids = linear_sum_assignment(iou, maximize=True)
        if min_iou:
            cost = iou[track_ids, other_ids]
            track_ids = track_ids[cost > min_iou]
            other_ids = other_ids[cost > min_iou]
        return track_ids, other_ids



def mask_pred_to_binary(x):
    idxs = torch.argmax(x, dim=0)
    y = torch.zeros_like(x)
    for i in range(len(x)):
        y[i, idxs==i] = 1
    return y

def mask_iou(a, b, eps=1e-7):
    a, b = a[:, None], b[None]
    overlap = (a * b) > 0
    union = (a + b) > 0
    return 1. * overlap.sum((2, 3)) / (union.sum((2, 3)) + eps)


def assign_masks(binary_masks, new_masks, pred_mask=None, min_iou=0.4):
    iou = mask_iou(binary_masks, new_masks)    
    iou = iou.cpu().numpy()
    rows, cols = linear_sum_assignment(iou, maximize=True)
    cost = iou[rows, cols]
    rows = rows[cost > min_iou]
    cols = cols[cost > min_iou]
    # cost = cost[cost > min_iou]
    # existing tracks without a matching detection
    unmatched_rows = sorted(set(range(len(binary_masks))) - set(rows))
    # new detections without a matching track
    unmatched_cols = sorted(set(range(len(new_masks))) - set(cols))
    # create indices for new tracks
    new_rows = torch.arange(len(unmatched_cols)) + len(binary_masks)

    # merge masks - create blank array with the right size
    n = len(binary_masks) + len(new_rows)
    full_masks = torch.zeros((n, *binary_masks.shape[1:]), device=binary_masks.get_device())
    new_masks = new_masks.float()

    # first add matches
    if len(rows):
        full_masks[rows] = new_masks[cols]
    # then for tracks that weren't matched, insert the xmem predictions
    if len(unmatched_rows):
        if pred_mask is None:
            pred_mask = binary_masks
        full_masks[unmatched_rows] = pred_mask[unmatched_rows]
    # for new detections without a track, insert with new track IDs
    if len(new_rows):
        full_masks[new_rows] = new_masks[unmatched_cols]

    # this is the track_ids corresponding to the input masks
    input_track_ids = [
        r for c, r in sorted(zip(
            (*cols, *unmatched_cols), 
            (*rows, *new_rows.tolist())))]

    return full_masks, input_track_ids, unmatched_rows, new_rows


def dilate_masks(mask, kernel, rel_area=0.01):
    # dilate masks whose overall area is below a certain value
    # this makes it easier for xmem to track those objects
    for i, size in enumerate(mask.sum((1, 2))):
        if size < rel_area * mask.shape[1] * mask.shape[2]:
            # print("dilating", labels[i], mask[i].shape)
            mask[i] = torch.as_tensor(
                cv2.dilate(mask[i].cpu().float().numpy(), kernel, iterations=1), 
                device=mask.device)
    return mask

def binarize_mask(prediction):
    # covert to binary mask
    if prediction.ndim == 4:
        prediction = prediction[0]
    pred_mask_int = torch.argmax(prediction, dim=0) - 1
    pred_ids = torch.unique(pred_mask_int)
    pred_ids = pred_ids[pred_ids >= 0]
    pred_mask = pred_ids[:, None, None] == pred_mask_int[None]
    return pred_mask