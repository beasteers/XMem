import cv2
import numpy as np
import torch
from xmem.inference.memory_manager import MemoryManager
from xmem.model.network import XMem
from xmem.model.aggregate import aggregate

from scipy.optimize import linear_sum_assignment

from xmem.util.tensor_util import pad_divide_by, unpad


class InferenceCore:
    def __init__(self, network:XMem, config):
        self.config = config
        self.network = network
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)

        self.clear_memory()
        self.all_labels = []
        self.next_label = 0

    def clear_memory(self):
        self.curr_ti = 0
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        self.memory = MemoryManager(config=self.config)

    def delete_object_id(self, object_id):
        index = self.all_labels.index(object_id)
        self.memory.delete_object_id(object_id, index)
        del self.all_labels[index]

    def update_config(self, config):
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        # if deep_update_every < 0, synchronize deep update with memory frame
        self.deep_update_sync = (self.deep_update_every < 0)
        self.memory.update_config(config)

    def set_all_labels(self, all_labels):
        if isinstance(all_labels, int):
            return self.add_labels(all_labels - len(self.all_labels))
        self.all_labels = all_labels
        self.next_label = max(max(self.all_labels), self.next_label)
        print("updated labels", self.all_labels)

    def add_labels(self, n_new):
        if n_new < 0: raise RuntimeError("not sure how to handle reducing label count.")
        labels = list(range(self.next_label, self.next_label + n_new))
        self.all_labels = self.all_labels + labels
        self.next_label += len(labels)
        print("added labels", labels)

    def step(self, image, mask=None, valid_labels=None, end=False):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0) # add the batch dimension
        if (mask is None or not mask.shape[0]) and not self.memory.work_mem.engaged():
            return torch.ones((1, *image.shape[1:])), []

        is_mem_frame = (
            (self.curr_ti-self.last_mem_ti >= self.mem_every) or 
            (mask is not None)
        ) and (not end)
        need_segment = (self.curr_ti > 0) and (
            (valid_labels is None) or 
            (len(self.all_labels) != len(valid_labels))
        )
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame) or  # synchronized
            (not self.deep_update_sync and self.curr_ti-self.last_deep_update_ti >= self.deep_update_every) # no-sync
        ) and (not end)
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (not end)

        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(
            image, 
            need_ek=(self.enable_long_term or need_segment), 
            need_sk=is_mem_frame)
        # f16: torch.Size([1, 1024, H/16, W/16])
        # f8:  torch.Size([1, 512,  H/8,  W/8])
        # f4:  torch.Size([1, 256,  H/4,  W/4])
        multi_scale_features = (f16, f8, f4)

        # segment the current frame is needed
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

        # associate masks with tracks using IoU
        track_ids = valid_labels
        if valid_labels is None:
            if mask is not None:
                if pred_prob_no_bg is not None:
                    pred_mask_no_bg = mask_pred_to_binary(pred_prob_with_bg)[1:]
                    mask, track_ids, unmatched_rows, new_rows = assign_masks(pred_mask_no_bg, mask, pred_prob_no_bg)
                    if len(new_rows):
                        # print("unmatched/new rows", unmatched_rows, new_rows, len(mask))
                        self.set_all_labels(len(mask))
                    track_ids = [self.all_labels[i] for i in track_ids]
                elif len(self.all_labels) != len(mask):
                    self.set_all_labels(len(mask))
                    track_ids = self.all_labels

        # use the input mask if any
        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)

            if pred_prob_no_bg is not None:
                # convert all masks overlapping with ground truth mask to zero
                pred_prob_no_bg[:, (mask.sum(0) > 0.5)] = 0
                # shift by 1 because mask/pred_prob_no_bg do not contain background
                mask = mask.type_as(pred_prob_no_bg)
                if valid_labels is not None:
                    shift_by_one_non_labels = [i for i in range(pred_prob_no_bg.shape[0]) if (i+1) not in valid_labels]
                    # non-labelled objects are copied from the predicted mask
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]

            # add background back in and convert using softmax
            pred_prob_with_bg = aggregate(mask, dim=0)
            # for i, m in enumerate(mask_pred_to_binary(pred_prob_with_bg)):
            #     cv2.imwrite(f'seg2{i}.jpg', m.cpu().int().numpy()*255)

            # also create new hidden states
            self.memory.create_hidden_state(len(self.all_labels), key)

        # save as memory if needed
        if is_mem_frame:
            # hidden: torch.Size([1, 3, 64, 2, 2])
            # image segmentation to memory value
            value, hidden = self.network.encode_value(
                image, f16, self.memory.get_hidden(), 
                pred_prob_with_bg[1:].unsqueeze(0), 
                is_deep_update=is_deep_update)
            # save value in memory
            self.memory.add_memory(
                key, shrinkage, value, self.all_labels, 
                selection=selection if self.enable_long_term else None)
            self.last_mem_ti = self.curr_ti

            if is_deep_update:
                self.memory.set_hidden(hidden)
                self.last_deep_update_ti = self.curr_ti

        self.curr_ti += 1
        # pred_prob_with_bg: [1 + num_objects, H, W]
        return unpad(pred_prob_with_bg, self.pad), track_ids



def mask_pred_to_binary(x):
    idxs = torch.argmax(x, dim=0)
    y = torch.zeros_like(x)
    for i in range(len(x)):
        y[i, idxs==i] = 1
    return y

def mask_iou(a, b, eps=1e-7):
    # print(a.shape, a.dtype, b.shape, b.dtype)
    # for i in range(a.shape[0]):
    #     cv2.imwrite(f'a{i}.jpg', a[i].cpu().numpy()*255)
    # for i in range(b.shape[0]):
    #     cv2.imwrite(f'b{i}.jpg', b[i].cpu().numpy()*255)
    # a, b: [num_objects, H, W]
    a, b = a[:, None], b[None]
    overlap = (a * b) > 0
    union = (a + b) > 0
    # print(a.min(), b.min(), a.float().mean(), b.float().mean())
    # print(union.min(), union.max())

    # for i in range(overlap.shape[0]):
    #     for j in range(overlap.shape[1]):
    #         cv2.imwrite(f'overlap{i}{j}.jpg', overlap[i, j].cpu().numpy()*255)
    #         cv2.imwrite(f'union{i}{j}.jpg', union[i, j].cpu().numpy()*255)
    # input()
    return 1. * overlap.sum((2, 3)) / (union.sum((2, 3)) + eps)


def assign_masks(binary_masks, new_masks, pred_mask, min_iou=0.4):
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
    # for r, c in zip(rows, cols):
    #     cv2.imwrite(f'ims/match-{r}{c}.jpg', np.concatenate([
    #         binary_masks[r].cpu().numpy()*255, 
    #         pred_mask[r].cpu().numpy()*255, 
    #         new_masks[c].cpu().numpy()*255
    #     ], 1))
    # # input()
    # print("unmatched cols", unmatched_cols)

    # create a mask combining everything
    new_rows = torch.arange(len(unmatched_cols)) + len(binary_masks)
    n = len(binary_masks) + len(new_rows)
    full_masks = torch.zeros((n, *binary_masks.shape[1:]), device=binary_masks.get_device())
    new_masks = new_masks.float()
    if len(rows):
        full_masks[rows] = new_masks[cols]
    if len(unmatched_rows):
        full_masks[unmatched_rows] = pred_mask[unmatched_rows]
    if len(new_rows):
        full_masks[new_rows] = new_masks[unmatched_cols]

    object_ids = [
        r for c, r in sorted(zip(
            (*cols, *unmatched_cols), 
            (*rows, *new_rows.tolist())))]
    return full_masks, object_ids, unmatched_rows, new_rows