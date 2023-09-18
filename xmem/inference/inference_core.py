import cv2
import numpy as np
import torch
import torch.nn.functional as F
import logging
import functools

from xmem.inference.track import Track
from xmem.inference.memory_manager import MemoryManager
from xmem.model.network import XMem as XMemModel
from xmem.model.aggregate import aggregate
from xmem.dataset.range_transform import im_normalization as norm
from xmem.util.tensor_util import pad_divide_by, unpad
from .config import get_config
from ..checkpoint import ensure_checkpoint
from torchvision.ops import masks_to_boxes
from .assignment import assign_masks, iou_assignment

from scipy.optimize import linear_sum_assignment


log = logging.getLogger('XMem')

device = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else 
    'cpu'
)

class XMem(torch.nn.Module):
    def __init__(self, config, checkpoint_path=None, map_location=device, model=None, Track=Track):
        super().__init__()
        config = get_config(config)
        if model is None:
            checkpoint_path = checkpoint_path or ensure_checkpoint()
            model = XMemModel(config, checkpoint_path, map_location)
            model.eval()

        self.model = model
        self.config = config
        self._device_param = torch.nn.Parameter(torch.empty(0))

        # box tracks
        self.Track = Track

        self._set_config(config)
        self.clear_memory()

    def clear_memory(self, reset_index=True):
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

    def delete_track(self, track_id):
        self.memory.delete_track(track_id)
    delete_object_id = delete_track

    def _set_config(self, config):
        # memory
        self._mem_every = config['mem_every']
        self._deep_update_every = config['deep_update_every']
        self._enable_long_term = config['enable_long_term']
        self._deep_update_sync = (self._deep_update_every < 0)  # synchronize deep update with memory frame

    @property
    def tracks(self) -> dict[Track]:
        return self.memory.tracks
    
    @property
    def track_ids(self) -> list[str]:
        return self.memory.track_ids

    def forward(self, image, mask=None, input_track_ids=None, no_update=False, binarize_mask=True, only_confirmed=False):
        '''Track object segmentations. This should be called on sequential frames. To reset memory, do `model.clear_memory()`

        Arguments:
            image (torch.Tensor): The image. Either a normalized Tensor (3, H, W) or a raw image np.array (H, W, 3).
            mask (torch.Tensor|None): The detection of ground truth mask used for tracking. size: (M, H, W)
            input_track_ids (np.ndarray): If you already know the track ID of a detection, you can feed it in here. 
            no_update (bool): Don't update memory this round.
            binarize_mask (bool): Whether to calculate argmax on the final mask. If False, the first channel of pred_mask will be background.
            only_confirmed (bool): Whether to drop tentative tracks or not.
        
        Returns:
            pred_mask: The predicted segmentations. (N, H, W) if binarize_mask else (1 + N, H, W) - includes background score.
            track_ids (np.ndarray): The N track IDs (one per pred_mask).
            input_track_ids (np.ndarray|None): The M track IDs (one per mask). Only returned if mask is provided, otherwise None.
        '''
        
        # ----------------------------- preprocess image ----------------------------- #

        image, pad = self._preprocess_image(image)

        # -------------------- check if there's anything to track -------------------- #

        # exit early if there is nothing to track
        if self._is_inactive(mask):
            return self._empty_detections(image, mask, binarize_mask=binarize_mask)

        # -------------------------- check memory conditions ------------------------- #

        is_mem_frame = not no_update and self._is_mem_frame(mask)
        is_deep_update = not no_update and self._is_deep_update(mask)
        need_segment = self._needs_segment(input_track_ids)

        # ---------------------------- encode and segment ---------------------------- #

        # encode image
        key, shrinkage, selection, multi_scale_features = self._encode_image(
            image, need_segment, is_mem_frame)

        # predict, segment image
        pred_prob_with_bg = None
        if need_segment:
            # predict segmentation mask for current timestamp
            pred_prob_with_bg = self._segment(
                key, selection, multi_scale_features, 
                is_normal_update=is_mem_frame and not is_deep_update)
            # pred_prob_with_bg: [1 + num_objects, H, W]

        # handle external segmentation mask
        valid_track_ids = input_track_ids
        if mask is not None:
            mask = F.pad(mask, pad)

            # match object detections with minimum IoU
            mask, input_track_ids = self._match_detection_masks(
                mask, pred_prob_with_bg, valid_track_ids, key)
            mask, pred_prob_with_bg = self._valid_mask_predictions(
                mask, pred_prob_with_bg, valid_track_ids)

        # ------------------------- update memory and outputs ------------------------ #

        # prepare outputs
        pred_mask = unpad(pred_prob_with_bg, pad)  # reverse padding
        track_ids = np.asarray(self.memory.track_ids)
        if input_track_ids is not None:
            input_track_ids = np.asarray(input_track_ids)

        # save to memory
        if is_mem_frame:
            self._update_memory(
                image, key, shrinkage, selection, multi_scale_features, 
                pred_prob_with_bg, is_deep_update)

        # handle track additions and deletions
        if input_track_ids is not None:
            pred_mask, track_ids = self._update_tracks(pred_mask, track_ids, input_track_ids)

        # ---------------------------- postprocess outputs --------------------------- #

        # convert probabilities to binary masks using argmax
        if binarize_mask:
            pred_mask, track_ids = self._binarize_mask(pred_mask, track_ids)
            # pred_mask: [num_objects, H, W] if binarize_mask else [1 + num_objects, H, W]

        # only return tracks that have been confirmed by multiple detections
        if only_confirmed:
            pred_mask, track_ids = self._filter_confirmed(pred_mask, confirmed)

        self.curr_it += 1
        return pred_mask, track_ids, input_track_ids
    
    def assign_masks(self, pred_prob_with_bg, mask):
        mask, input_track_ids, unmatched_rows, new_rows = assign_masks(
            pred_prob_with_bg, mask,
            self.config['min_iou'],
            self.config['allow_create'])
        return mask, input_track_ids, unmatched_rows, new_rows

    # -------------------------------- Conditions -------------------------------- #

    def _is_inactive(self, mask):
        # does xmem have anything to do?
        return not self.memory.work_mem.engaged() and (mask is None or not mask.shape[0])
    
    def _is_mem_frame(self, mask):
        # should we update memory?
        return (
            # update every N steps
            (self.curr_it - self.last_mem_it >= self._mem_every) or 
            # or when provided with new masks
            (mask is not None))

    def _is_deep_update(self, is_mem_frame):
        # should we do a deep memory update?
        return (
            # do a deep update on memory update (if synced)
            is_mem_frame
            if self._deep_update_sync else 
            self.curr_it - self.last_deep_update_it >= self._deep_update_every
            # do a deep update every N steps (if synced)
        )

    def _needs_segment(self, valid_track_ids):
        # do we need to compute segmentation masks?
        return self.memory.work_mem.engaged() and (
            # if the user doesn't provide assigned masks
            valid_track_ids is None or 
            # or not for for every track
            len(self.memory.track_ids) != len(valid_track_ids))

    # ----------------------------- algorithm steps: ----------------------------- #

    def _preprocess_image(self, image):
        '''prepare image for model.'''
        # image: [3, H, W]
        # mask: [num_objects, H, W] or None
        if not torch.is_tensor(image):
            image = norm(torch.from_numpy(image.transpose(2, 0, 1)).float().to(self._device_param.device)/255)
        image, pad = pad_divide_by(image, 16)  # image should be divisible by 16
        image = image.unsqueeze(0) # add the batch dimension
        return image, pad

    def _empty_detections(self, image, mask, binarize_mask=True):
        '''when there's no tracks and no detections, return empty masks.'''
        mask = torch.ones((0 if binarize_mask else 1, *image.shape[-2:]), device=image.device)
        input_track_ids = None if mask is None else np.array([])
        return mask, np.array([]), input_track_ids

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
        if is_normal_update:
            self.memory.set_hidden(hidden)
        return pred_prob_with_bg

    def _match_detection_masks(self, mask, pred_prob_with_bg, valid_track_ids=None, key=None):
        '''Match object detections with minimum IoU'''
        input_track_ids = valid_track_ids
        if valid_track_ids is None:
            if pred_prob_with_bg is None:
                assert not self.memory.work_mem.engaged()
                self.memory.update_track_ids(len(mask), key)
                input_track_ids = self.memory.track_ids
            else:
                mask, input_track_ids, unmatched_rows, new_rows = self.assign_masks(
                    pred_prob_with_bg, mask)
                if len(new_rows):
                    # print("unmatched/new rows", unmatched_rows, new_rows, len(mask))
                    self.memory.update_track_ids(len(mask), key)
                track_ids = self.memory.track_ids
                input_track_ids = [track_ids[i] for i in input_track_ids]

        assert (mask is None) == (input_track_ids is None)
        return mask, input_track_ids
    
    def _valid_mask_predictions(self, mask, pred_prob_with_bg, valid_track_ids):
        '''Combine predicted and detected masks.'''
        pred_prob_no_bg = pred_prob_with_bg[1:]
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
        return mask, pred_prob_with_bg
    
    def _update_memory(self, image, key, shrinkage, selection, multi_scale_features, pred_prob_with_bg, is_deep_update):
        '''Add features to memory and possibly do a full update hidden state.'''
        # hidden: torch.Size([1, 3, 64, 2, 2])
        # image segmentation to memory value
        torch.cuda.synchronize()
        value, hidden = self.model.encode_value(
            image, multi_scale_features[0], self.memory.get_hidden(), 
            pred_prob_with_bg[1:].unsqueeze(0), 
            is_deep_update=is_deep_update)
        torch.cuda.synchronize()
        # save value in memory
        self.memory.add_memory(
            key, shrinkage, value, 
            selection=selection if self._enable_long_term else None)
        self.last_mem_it = self.curr_it
        torch.cuda.synchronize()

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

        # remove batch dimension
        if pred_mask.ndim == 4:
            pred_mask = pred_mask[0]

        # get the winning index
        pred_mask_int = torch.argmax(pred_mask, dim=0) - 1
        pred_ids = torch.arange(len(pred_mask)-1, device=pred_mask.device)
        pred_mask = pred_ids[:, None, None] == pred_mask_int[None]

        # filter empty
        present = pred_mask.any(2).any(1)#.cpu().numpy()
        pred_mask = pred_mask[present]
        pred_ids = pred_ids[present]

        # what should the output format be?
        track_ids = track_ids[pred_ids.cpu().numpy()]
        return pred_mask, track_ids

    def _filter_confirmed(self, pred_mask, track_ids):
        confirmed = np.array([self.memory.tracks[t].is_confirmed() for t in track_ids], dtype=bool)
        pred_mask = pred_mask[confirmed]
        track_ids = track_ids[confirmed]
        return pred_mask, track_ids


    # -------------------------- User convenience utils -------------------------- #

    def match_iou(self, xmem_mask, other_mask, min_iou=0.4):
        track_ids, other_ids = self.iou_assignment(xmem_mask, other_mask, min_iou)
        track_ids = np.array(self.memory.track_ids, dtype=int)[track_ids]
        return track_ids, other_ids
    
    @staticmethod
    def iou_assignment(first_mask, other_mask, min_iou=0.4):
        return iou_assignment(first_mask, other_mask, min_iou)
    
    @staticmethod
    def masks_to_boxes(masks):
        return masks_to_boxes(masks)

    @staticmethod
    def dilate_masks(masks, kernel=5, rel_area=0.01):
        return dilate_masks(masks, kernel, rel_area)



@functools.lru_cache(3)
def _get_kernel(kernel_size):
    if isinstance(kernel_size, np.ndarray):
        return kernel_size
    return np.ones((kernel_size, kernel_size), dtype=np.int32)


def dilate_masks(mask, kernel=5, rel_area=0.01):
    '''dilate masks whose overall area is below a certain value. 
    this makes it easier for xmem to track those objects (e.g. toothpicks).
    '''
    kernel = _get_kernel(kernel)
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
