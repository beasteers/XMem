import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from torchvision.ops import masks_to_boxes
from IPython import embed

import logging
log = logging.getLogger(__name__)


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

# def mask_iou_a_engulf_b(a, b, eps=1e-7):
#     a, b = a[:, None], b[None]
#     overlap = (a * b) > 0
#     return 1. * overlap.sum((2, 3)) / (b.sum((2, 3)) + eps)

def box_ioa(xx, yy, method='union', eps=1e-7):
    # Calculate the area of intersection
    intersection_area = (
        torch.clamp(torch.minimum(xx[:, None, 2], yy[None, :, 2]) - torch.maximum(xx[:, None, 0], yy[None, :, 0]), 0) * 
        torch.clamp(torch.minimum(xx[:, None, 3], yy[None, :, 3]) - torch.maximum(xx[:, None, 1], yy[None, :, 1]), 0)
    )

    # Calculate the area of each bounding box
    area_xx = (xx[:, 2] - xx[:, 0]) * (xx[:, 3] - xx[:, 1])
    area_yy = (yy[:, 2] - yy[:, 0]) * (yy[:, 3] - yy[:, 1])
    # if method == 'union':
    #     base_area = area_xx[:, None] + area_yy[None] - intersection_area
    # elif method == 'min':
    #     base_area = torch.minimum(area_xx[:, None], area_yy[None])
    # elif method == 'max':
    #     base_area = torch.maximum(area_xx[:, None], area_yy[None])
    # else:
    #     raise ValueError(f"Invalid box IoU method: {method}")
    union_area = area_xx[:, None] + area_yy[None] - intersection_area
    min_area = torch.minimum(area_xx[:, None], area_yy[None])

    # Calculate IoU (Intersection over Area)
    return intersection_area / (union_area + eps), intersection_area / (min_area + eps)

def box_center_dist(xx, yy, eps=1e-7):
    # Calculate the center coordinates of each bounding box
    center_xx = torch.stack([(xx[:, 0] + xx[:, 2]) / 2, (xx[:, 1] + xx[:, 3]) / 2], dim=1)
    center_yy = torch.stack([(yy[:, 0] + yy[:, 2]) / 2, (yy[:, 1] + yy[:, 3]) / 2], dim=1)
    center_distance = torch.norm(center_xx[:, None, :] - center_yy[None, :, :], dim=2)

    # Calculate the minimum width and height for each pair of boxes
    width = torch.maximum(xx[:, None, 2] - xx[:, None, 0], yy[:, 2] - yy[:, 0])
    height = torch.maximum(xx[:, None, 3] - xx[:, None, 1], yy[:, 3] - yy[:, 1])
    base_dist = torch.minimum(width, height)

    # Calculate box center distance divided by the minimum width/height
    return center_distance / (base_dist + eps)


# def asymmetric_nms(boxes, scores, iou_threshold=0.99):
#     nn=len(boxes)
#     # Sort boxes by their confidence scores in descending order
#     area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     indices = torch.argsort(area, descending=True)
#     # indices = np.argsort(scores)[::-1]
#     boxes = boxes[indices]
#     scores = scores[indices]

#     selected_indices = []
#     overlap_indices = []
#     while len(boxes) > 0:
#         # Pick the box with the highest confidence score
#         b = boxes[0]
#         selected_indices.append(indices[0])

#         # Calculate IoU between the picked box and the remaining boxes
#         zero = torch.tensor([0], device=boxes.device)
#         intersection_area = (
#             torch.maximum(zero, torch.minimum(b[2], boxes[1:, 2]) - torch.maximum(b[0], boxes[1:, 0])) * 
#             torch.maximum(zero, torch.minimum(b[3], boxes[1:, 3]) - torch.maximum(b[1], boxes[1:, 1]))
#         )
#         smaller_box_area = torch.minimum(area[0], area[1:])
#         iou = intersection_area / (smaller_box_area + 1e-7)

#         # Filter out boxes with IoU above the threshold

#         overlap_indices.append(indices[torch.where(iou > iou_threshold)[0]])
#         filtered_indices = torch.where(iou <= iou_threshold)[0]
#         indices = indices[filtered_indices + 1]
#         boxes = boxes[filtered_indices + 1]
#         scores = scores[filtered_indices + 1]
#         area = area[filtered_indices + 1]

#     selected_indices = (
#         torch.stack(selected_indices) if selected_indices else 
#         torch.zeros([0], dtype=torch.int32, device=boxes.device))
#     print(nn, overlap_indices)
#     if nn>1 and input():embed()
#     return selected_indices, overlap_indices


def masks_to_boxes2(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    bounding_boxes = torch.zeros((masks.shape[0], 4), device=masks.device, dtype=torch.float)
    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        if len(x):
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)
    return bounding_boxes


def assign_masks(
        pred_prob_with_bg, new_masks, 
        label_cost=None, track_det_mask=None, 
        min_iou=0.4, min_box_iou=0.8, min_box_ioa=0.98, 
        max_center_dist=0.3, min_box_center_ioa=0.2, min_label_cost=0.5, **kw
    ):
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
    # get binary and probabilistic masks of xmem predictions
    binary_masks = mask_pred_to_binary(pred_prob_with_bg)[1:]
    pred_masks = pred_prob_with_bg[1:]  # drop background

    all_masks = new_masks# if other_masks is None else torch.cat([new_masks, other_masks], dim=0)
    if track_det_mask is not None:
        new_masks = all_masks[track_det_mask]
    
    # compute assignment
    cost = mask_iou(binary_masks, all_masks).cpu().numpy()
    tboxes = masks_to_boxes2(binary_masks)
    nboxes = masks_to_boxes2(all_masks)
    iou_cost, ioa_cost = box_ioa(tboxes, nboxes)
    ioa_cost = ioa_cost.cpu().numpy()
    iou_cost = iou_cost.cpu().numpy()
    if label_cost is not None:
        log.debug(f"label cost: {np.round(label_cost.cpu(), 2)}")
    # center_cost = box_center_dist(tboxes, nboxes).cpu().numpy()
    log.debug(f"mask cost: {np.round(cost, 2)}")
    log.debug(f"ioa cost: {np.round(ioa_cost, 2)}")
    log.debug(f"iou cost: {np.round(iou_cost, 2)}")
    
    rows, cols = linear_sum_assignment(cost, maximize=True)
    xcost = cost[rows, cols]
    xiou_cost = iou_cost[rows, cols]
    xioa_cost = ioa_cost[rows, cols]
    # xcenter_cost = center_cost[rows, cols]
    keep = (
        # has a high enough segmentation match
        (xcost > min_iou)
        
    )
    weak_keep = (
        # has high enough box match
        (xiou_cost > min_box_iou)
        # one box contained in another
        | (xioa_cost > min_box_ioa)
        # # box centers close
        # | ((xcenter_cost < max_center_dist) & (xioa_cost > min_box_center_ioa))
    )
    if label_cost is not None:
        xlabel_cost = label_cost[rows, cols].cpu().numpy()
        weak_keep &= xlabel_cost > min_label_cost
    keep |= weak_keep
    rows = rows[keep]
    cols = cols[keep]

    if track_det_mask is not None:
        # keep = (np.cumsum(track_det_mask.int()) - 1)[cols]
        # keep = np.arange(len(track_det_mask))[cols]
        c1,r1=cols, rows
        indices = np.where(track_det_mask)[0]
        index_mapping = np.zeros(len(track_det_mask), dtype=int)-1
        index_mapping[indices] = np.arange(len(indices))
        cols = index_mapping[cols]
        keep = cols >= 0
        rows = rows[keep]
        cols = cols[keep]

    return combine_masks(pred_masks, new_masks, rows, cols, **kw)


def combine_masks(pred_masks, new_masks, rows, cols, allow_create=True, join_method='override'):
    # existing tracks without a matching detection
    unmatched_rows = sorted(set(range(len(pred_masks))) - set(rows))
    # new detections without a matching track
    unmatched_cols = sorted(set(range(len(new_masks))) - set(cols))

    # ---------------------- Merge everything into one mask ---------------------- #

    # create indices for new tracks
    new_rows = torch.arange(len(unmatched_cols) if allow_create else 0) + len(pred_masks)
    # merge masks - create blank array with the right size
    n = len(pred_masks) + len(new_rows)
    full_masks = torch.zeros((n, *pred_masks.shape[1:]), device=pred_masks.get_device())
    new_masks = new_masks.float()

    # # first override matches
    if len(rows):
        if join_method == 'replace':  # trust detection masks
            full_masks[rows] = new_masks[cols]
        elif join_method == 'ignore':  # trust tracking masks
            full_masks[rows] = pred_masks[rows]
        elif join_method == 'max':  # take the maximum of the two masks XXX idk if this makes sense
            full_masks[rows] = torch.maximum(new_masks[cols], pred_masks[rows])
        elif join_method == 'min':  # take the minimum of the two masks XXX idk if this makes sense
            full_masks[rows] = torch.minimum(new_masks[cols], pred_masks[rows])
        elif join_method == 'mult':  # scale the likelihood of XMem using the detections [0.5-1.5]
            full_masks[rows] = (new_masks[cols] + 0.5) * pred_masks[rows]
        else:
            raise ValueError("Invalid mask join method")
    # then for tracks that weren't matched, insert the xmem predictions
    if len(unmatched_rows):
        full_masks[unmatched_rows] = pred_masks[unmatched_rows]
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


def iou_assignment(first_mask, other_mask, min_iou=0.4):
    iou = mask_iou(first_mask, other_mask)
    iou = iou.cpu().numpy() if isinstance(iou, torch.Tensor) else iou
    track_ids, other_ids = linear_sum_assignment(iou, maximize=True)
    if min_iou:
        cost = iou[track_ids, other_ids]
        track_ids = track_ids[cost > min_iou]
        other_ids = other_ids[cost > min_iou]
    return track_ids, other_ids

