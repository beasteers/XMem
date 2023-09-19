import torch
from scipy.optimize import linear_sum_assignment



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

def mask_iou_a_engulf_b(a, b, eps=1e-7):
    a, b = a[:, None], b[None]
    overlap = (a * b) > 0
    return 1. * overlap.sum((2, 3)) / (b.sum((2, 3)) + eps)


def assign_masks(pred_prob_with_bg, new_masks, min_iou=0.4, allow_create=True):
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
    
    # compute assignment
    cost = mask_iou(binary_masks, new_masks)    
    cost = cost.cpu().numpy()
    rows, cols = linear_sum_assignment(cost, maximize=True)
    xcost = cost[rows, cols]
    rows = rows[xcost > min_iou]
    cols = cols[xcost > min_iou]

    # TODO: assign masks that are engulfed by an existing track? see merge_by_engulf in DEVA
    # iou_engulf = mask_iou_a_engulf_b()
    # # idk maybe it's not actually what we want
    return combine_masks(pred_masks, new_masks, rows, cols, allow_create=True)


def combine_masks(pred_masks, new_masks, rows, cols, allow_create=True):
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

    # first add matches
    if len(rows):
        full_masks[rows] = new_masks[cols]
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

