import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging

from xmem.inference.track import Track
from xmem.inference.memory_manager import MemoryManager
from xmem.model.network import XMem as XMemModel
from xmem.model.aggregate import aggregate
from xmem.util.tensor_util import pad_divide_by, unpad
from .config import get_config
from ..checkpoint import ensure_checkpoint
from torchvision.ops import masks_to_boxes

from scipy.optimize import linear_sum_assignment


log = logging.getLogger('XMem')

device = 'cuda'

class XMem(torch.nn.Module):
    next_label = 0
    def __init__(self, config, checkpoint_path=None, map_location=None, model=None, Track=Track):
        super().__init__()
        config = get_config(config)
        if model is None:
            checkpoint_path = checkpoint_path or ensure_checkpoint()
            model = XMemModel(config, checkpoint_path, map_location)
            model.eval()

        self.model = model
        self.config = config

        # box tracks
        self.Track = Track

        self._set_config(config)
        self.clear_memory()

    def clear_memory(self, reset_index=False):
        # memory
        self.curr_it = 0
        self.last_mem_it = 0
        if not self._deep_update_sync:
            self.last_deep_update_it = -self._deep_update_every
        
        track_offset = 0
        if not reset_index and hasattr(self, 'memory'):
            track_offset = self.memory.track_offset
        self.memory = MemoryManager(config=self.config, Track=self.Track, track_offset=track_offset)

    def update_config(self, config):
        self._set_config(config)
        self.memory.update_config(config)

    def delete_object_id(self, object_id):
        self.memory.delete_object_id(object_id)

    def _set_config(self, config):
        # memory
        self._mem_every = config['mem_every']
        self._deep_update_every = config['deep_update_every']
        self._enable_long_term = config['enable_long_term']
        self._deep_update_sync = (self._deep_update_every < 0)  # synchronize deep update with memory frame

        # a kernel to make tiny segmentatation maps a bit bigger (e.g. toothpicks)
        self._dilate_size_threshold = config.get('dilate_size_threshold')
        kernel_size = config.get('dilation_kernel_size') or 0
        self._dilation_kernel = np.ones((kernel_size, kernel_size), dtype=np.int32)

    def forward(self, image, mask=None, valid_track_ids=None, no_update=False, binarize_mask=True, only_confirmed=False, only_visible=True):
        
        # ----------------------------- preprocess image ----------------------------- #

        # image: [3, H, W]
        # mask: [num_objects, H, W] or None
        image, pad = pad_divide_by(image, 16)  # image should be divisible by 16
        image = image.unsqueeze(0) # add the batch dimension

        # --------------------------- check empty tracker: --------------------------- #

        # exit early if there is nothing to track
        engaged = self.memory.work_mem.engaged()
        if not engaged and (mask is None or not mask.shape[0]):
            input_track_ids = None if mask is None else np.array([])
            return (
                torch.ones((0 if binarize_mask else 1, *image.shape[-2:]), device=image.device), 
                np.array([]), 
                input_track_ids)

        # ------------------------- check memory conditions: ------------------------- #

        # should we update memory?
        is_mem_frame = not no_update and (
            # update every N steps
            (self.curr_it - self.last_mem_it >= self._mem_every) or 
            # or when provided with new masks
            (mask is not None)
        )

        # do we need to compute segmentation masks?
        need_segment = engaged and (
            # if the user doesn't provide assigned masks
            valid_track_ids is None or 
            # or not for for every track
            len(self.memory.track_ids) != len(valid_track_ids)
        )

        # should we do a deep memory update?
        is_deep_update = not no_update and (
            # do a deep update on memory update (if synced)
            is_mem_frame
            if self._deep_update_sync else 
            self.curr_it - self.last_deep_update_it >= self._deep_update_every
            # do a deep update every N steps (if synced)
        )

        # ------------------------ encode, segment, and match: ----------------------- #

        # encode image
        key, shrinkage, selection, multi_scale_features = self._encode_image(
            image, need_segment, is_mem_frame
        )

        # predict, segment image
        pred_prob_no_bg = pred_prob_with_bg = None
        if need_segment:
            # predict segmentation mask for current timestamp
            pred_prob_with_bg, pred_prob_no_bg = self._segment(
                key, selection, multi_scale_features, 
                is_normal_update=is_mem_frame and not is_deep_update
            )

        # handle external segmentation mask
        input_track_ids = valid_track_ids
        if mask is not None:
            mask = F.pad(mask, pad)

            # match object detections with minimum IoU
            mask, input_track_ids = self._match_detection_masks(
                mask, pred_prob_no_bg, pred_prob_with_bg, valid_track_ids, key
            )
            mask, pred_prob_no_bg, pred_prob_with_bg = self._valid_mask_predictions(
                mask, pred_prob_no_bg, pred_prob_with_bg, valid_track_ids
            )

        # ------------------------ update memory and outputs: ------------------------ #

        # prepare outputs
        pred_mask = unpad(pred_prob_with_bg, pad)  # reverse padding
        track_ids = np.asarray(self.memory.track_ids)
        input_track_ids = np.asarray(input_track_ids) if input_track_ids is not None else None

        # save to memory
        if is_mem_frame:
            self._update_memory(
                image, key, shrinkage, selection, multi_scale_features, 
                pred_prob_with_bg, is_deep_update)

        # handle track additions and deletions
        if input_track_ids is not None:
            pred_mask, track_ids = self._update_tracks(pred_mask, track_ids, input_track_ids)

        # convert probabilities to binary masks using argmax
        if binarize_mask:
            pred_mask, track_ids = self._binarize_mask(pred_mask, track_ids)

        # only return tracks that have been confirmed by multiple detections
        if only_confirmed:
            confirmed = np.array([self.memory.tracks[t].is_confirmed() for t in track_ids], dtype=bool)
            pred_mask = pred_mask[confirmed]
            track_ids = track_ids[confirmed]

        # filter out segmentation masks where the entire mask is zeros
        if only_visible:
            visible_tracks = pred_mask.any(1).any(1).cpu().numpy()
            pred_mask = pred_mask[visible_tracks]
            track_ids = track_ids[visible_tracks]

        self.curr_it += 1
        # pred_prob_with_bg: [1 + num_objects, H, W]
        # pred_mask: [num_objects, H, W] if binarize_mask else [1 + num_objects, H, W]
        return pred_mask, track_ids, input_track_ids
    



    # ----------------------------- algorithm steps: ----------------------------- #


    def _encode_image(self, image, need_segment, is_mem_frame):
        '''Encode image features'''
        key, shrinkage, selection, f16, f8, f4 = self.model.encode_key(
            image, 
            need_ek=(self._enable_long_term or need_segment), 
            need_sk=is_mem_frame)
        # f16: torch.Size([1, 1024, H/16, W/16])
        # f8:  torch.Size([1, 512,  H/8,  W/8])
        # f4:  torch.Size([1, 256,  H/4,  W/4])
        multi_scale_features = (f16, f8, f4)
        return key, shrinkage, selection, multi_scale_features

    def _segment(self, key, selection, multi_scale_features, is_normal_update):
        '''Compute the segmentation from memory'''
        # shape [1, ?]
        memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)
        # get track masks
        hidden, _, pred_prob_with_bg = self.model.segment(
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
        return pred_prob_with_bg, pred_prob_no_bg

    def _match_detection_masks(self, mask, pred_prob_no_bg, pred_prob_with_bg, valid_track_ids=None, key=None):
        '''Match object detections with minimum IoU'''
        input_track_ids = valid_track_ids
        # increase the size of very small masks to make tracking easier
        if self._dilate_size_threshold:
            mask = dilate_masks(mask, self._dilation_kernel, self._dilate_size_threshold)

        if valid_track_ids is None:
            if pred_prob_no_bg is None:
                assert not self.memory.work_mem.engaged()
                self.memory.update_track_ids(len(mask), key)
                input_track_ids = self.memory.track_ids
            else:
                pred_mask_no_bg = mask_pred_to_binary(pred_prob_with_bg)[1:]
                mask, input_track_ids, unmatched_rows, new_rows = assign_masks(
                    pred_mask_no_bg, mask, pred_prob_no_bg,
                    self.config['min_iou'],
                    self.config['allow_create'])
                if len(new_rows):
                    # print("unmatched/new rows", unmatched_rows, new_rows, len(mask))
                    self.memory.update_track_ids(len(mask), key)
                track_ids = self.memory.track_ids
                input_track_ids = [track_ids[i] for i in input_track_ids]

        assert (mask is None) == (input_track_ids is None)
        return mask, input_track_ids
    
    def _valid_mask_predictions(self, mask, pred_prob_no_bg, pred_prob_with_bg, valid_track_ids):
        '''Combine predicted and detected masks.'''
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
        return mask, pred_prob_no_bg, pred_prob_with_bg
    
    def _update_memory(self, image, key, shrinkage, selection, multi_scale_features, pred_prob_with_bg, is_deep_update):
        '''Add features to memory and possibly do a full update hidden state.'''
        # hidden: torch.Size([1, 3, 64, 2, 2])
        # image segmentation to memory value
        value, hidden = self.model.encode_value(
            image, multi_scale_features[0], self.memory.get_hidden(), 
            pred_prob_with_bg[1:].unsqueeze(0), 
            is_deep_update=is_deep_update)
        # save value in memory
        self.memory.add_memory(
            key, shrinkage, value, 
            selection=selection if self._enable_long_term else None)
        self.last_mem_it = self.curr_it

        if is_deep_update:
            self.memory.set_hidden(hidden)
            self.last_deep_update_it = self.curr_it

        log.debug("%s memory update %s: \n%s", 'deep' if is_deep_update else 'normal', self.curr_it, self.memory)

    def _update_tracks(self, pred_mask, track_ids, input_track_ids):
        '''Bookkeeping track detections'''
        track_ids = np.asarray(track_ids)
        deleted = self.memory.update_tracks(input_track_ids, self.curr_it)
        keep = np.array([True]+[i not in deleted for i in track_ids])
        pred_mask = pred_mask[keep]
        track_ids = track_ids[keep[1:]]
        return pred_mask, track_ids
    
    def _binarize_mask(self, pred_mask, track_ids):
        '''Convert logits to argmax binary mask.'''
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
        return pred_mask, track_ids


    # -------------------------- User convenience utils -------------------------- #

    def match_iou(self, xmem_mask, other_mask, min_iou=0.4):
        track_ids, other_ids = self.iou_assignment(xmem_mask, other_mask, min_iou)
        track_ids = np.array(self.memory.track_ids, dtype=int)[track_ids]
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
    
    @staticmethod
    def masks_to_boxes(masks):
        return masks_to_boxes(masks)



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


def assign_masks(binary_masks, new_masks, pred_mask=None, min_iou=0.4, allow_create=True):
    '''Assign XMem predicted masks to user-provided masks.
    
    Arguments:
        binary_masks (torch.Tensor): The binary masks given by XMem.
        new_masks (torch.Tensor): The binary masks given by you.
        pred_mask (torch.Tensor): The probabilistic masks given by XMem.
        min_iou (float): The minimum IoU allowed for mask assignment.
    
    Returns:
        full_masks (torch.Tensor): The merged masks.
        input_track_ids (list): The track indices corresponding to each input mask. If allow_create=False, these values can be None.
        unmatched_tracks (list): Tracks that did not have associated matches.
        new_tracks (torch.Tensor): Track indices that were added.

    NOTE: returned track IDs correspond to the track index in ``binary_mask``. If you have
          another index of tracks (e.g. if you manage track deletions) you need to re-index
          those externally.
    '''
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
    new_rows = torch.arange(len(unmatched_cols) if allow_create else 0) + len(binary_masks)

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
    new_rows_list = new_rows.tolist()
    if not allow_create:
        new_rows_list = [None]*len(unmatched_cols)
    input_track_ids = [
        r for c, r in sorted(zip(
            (*cols, *unmatched_cols), 
            (*rows, *new_rows_list)))]
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
