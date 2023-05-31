'''XMem + Detic + Deep-Sort

usage: python demo.py ../R0-P03_03.mp4 --vocab '["peanut butter jar", "jelly jar", "plate", "cutting board", "paper towel", "tortilla, white circular flat", "burrito", "toothpicks", "floss", "person", "knife"]' --fps-down 3
'''
import time
from collections import defaultdict, Counter
import cv2
import tqdm
import numpy as np
import torch
import supervision as sv

from xmem.inference import InferenceCore, XMem
from xmem.inference.interact.interactive_utils import image_to_torch, torch_prob_to_numpy_mask, overlay_davis
from torchvision.ops import masks_to_boxes
import deep_sort



device = 'cuda'

# default configuration
xmem_config = {
    'top_k': 20,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 64,
    'min_mid_term_frames': 3,
    'max_mid_term_frames': 6,
    'max_long_term_elements': 10000,
}


@torch.no_grad()
def main(src, vocab='lvis', detect_every=1, skip_frames=30, fps_down=1, size=480):
    from detic import Detic
    from ptgprocess.util import draw_boxes

    # NOTE: xmem rounds image sizes to a factor of 16
    WH = (int(720/480*size)//16*16, int(size)//16*16)

    # object detector
    detic_model = Detic(['cat'], conf_threshold=0.5, masks=True).to(device)
    detic_model.set_vocab(vocab)
    print(detic_model.labels)

    # box tracker
    ds_tracker = deep_sort.Tracker(deep_sort.NearestNeighbor('cosine', 0.2), max_age=None)

    # mask tracker
    network = XMem(xmem_config).eval().to(device)
    processor = InferenceCore(network, config=xmem_config)

    # object label counter (dict of object_track_id -> {plate: 25, cutting_board: 1})
    object_labels = defaultdict(lambda: Counter())

    # a kernel to make tiny segmentatation maps a bit bigger (e.g. toothpicks)
    kernel = np.ones((5, 5), dtype=np.int32)

    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    video_info.width, video_info.height = WH
    video_info.width *= 2
    og_fps = video_info.fps
    video_info.fps /= fps_down
    t_obs = 0
    with sv.VideoSink(target_path='xmem_output.mp4', video_info=video_info) as s:
        for i, frame in enumerate(tqdm.tqdm(sv.get_video_frames_generator(src))):
            if i < skip_frames:
                continue
            if i%fps_down: continue

            # resize the input frame and make a copy for the output video so we can draw on it
            frame = cv2.resize(frame, WH)
            out_frame = frame.copy()

            # # debug xmem memory every once in a while
            # if not i % 200:
            #     print(processor.memory)

            ######
            # Run detic to predict semantic bounding boxes based on text labels
            ######

            labels = mask = None
            if not i % int(detect_every*og_fps) or i == skip_frames:
                outputs = detic_model(frame)["instances"]
                mask = outputs.pred_masks.int()
                mask1=torch.argmax(torch.cat([torch.zeros(1, *mask.shape[1:]), mask.cpu()]), dim=0)
                out_frame2 = overlay_davis(out_frame.copy(), mask1)

                xywh = outputs.pred_boxes.tensor.cpu().numpy()
                cls_ids = outputs.pred_classes.cpu().numpy()
                scores = outputs.scores.cpu().numpy()
                features = outputs.clip_features.cpu().numpy()
                labels = detic_model.labels[cls_ids]

                # mask_f = mask.float()
                for i, size in enumerate(mask.sum((1, 2))):
                    if size < 0.01 * mask.shape[1] * mask.shape[2]:
                        # print("dilating", labels[i], mask[i].shape)
                        mask[i] = torch.as_tensor(cv2.dilate(mask[i].cpu().float().numpy(), kernel, iterations=1), device=device)

                # matches, unmatched_tracks, unmatched_detections = ds_tracker._match([])
                t_obs += 1
                ds_tracker.predict(t_obs)
                detections = [
                    deep_sort.Detection(xy, conf, z)
                    for xy, conf, z in zip(xywh, scores, features)
                ]

            ######
            # Run object tracking
            ######

            frame_torch, _ = image_to_torch(frame, device=device)
            prediction, track_ids = processor.step(frame_torch, mask)

            # update deep-sort and label counter with tracking results
            all_labels = list(processor.all_labels)
            if labels is not None:
                if track_ids is None:
                    track_ids = processor.all_labels

                # update matches in deep sort - use deep-sort to delete objects that are bad matches across multiple frames
                matches = [(ti, di) for di, ti in enumerate(track_ids)]
                deleted = ds_tracker._update(detections, t_obs, matches)
                deleted and print('deleting', set(deleted))
                for i in deleted:
                    processor.delete_object_id(i)

                # update object counts
                for ((ti, di), l) in zip(matches, labels):
                    object_labels[ti].update((l,))

                draw_boxes(out_frame2, xywh, [
                    [f'{i} {l}'] 
                    for i, l in zip(track_ids, detic_model.labels[cls_ids])
                ])

            ######
            # Draw segmentation masks from tracker along with common object labels
            ######

            if len(prediction):
                # draw mask
                pred_mask = torch_prob_to_numpy_mask(prediction)
                out_frame = overlay_davis(out_frame, pred_mask)

                # convert masks to bounding boxes and draw
                if pred_mask.any():
                    ids=np.unique(pred_mask)[1:]
                    ids = [i for i in ids if all_labels[i-1] in ds_tracker.tracks and ds_tracker.tracks[all_labels[i-1]].is_confirmed()]
                    m=torch.zeros((len(ids), *pred_mask.shape))
                    for i in range(len(ids)):
                        m[i, pred_mask == ids[i]] = 1
                    xywh = masks_to_boxes(m)
                    draw_boxes(out_frame, xywh, [
                        [f'{i} {l} {c}' for l, c in object_labels[i].most_common(3)] 
                        for i in (all_labels[i-1] for i in ids)
                    ])

            s.write_frame(np.concatenate([out_frame, out_frame2], axis=1))



# bounding box drawing utils

import itertools
def draw_boxes(im, boxes, labels=None, color=(0,255,0), size=1, text_color=(0, 0, 255), spacing=3):
    boxes = np.asarray(boxes).astype(int)
    color = np.asarray(color).astype(int)
    color = color[None] if color.ndim == 1 else color
    labels = itertools.chain([] if labels is None else labels, itertools.cycle(['']))
    for xy, label, c in zip(boxes, labels, itertools.cycle(color)):
        im = cv2.rectangle(im, xy[:2], xy[2:4], tuple(c.tolist()), 2)
        if label:
            if isinstance(label, list):
                im, _ = draw_text_list(im, label, 0, tl=xy[:2] + spacing, space=40, color=text_color)
            else:
                im = cv2.putText(im, label, xy[:2] - spacing, cv2.FONT_HERSHEY_SIMPLEX, im.shape[1]/1400*size, text_color, 1)
    return im


def draw_text_list(img, texts, i=-1, tl=(10, 50), scale=0.4, space=50, color=(255, 255, 255), thickness=1):
    for i, txt in enumerate(texts, i+1):
        cv2.putText(
            img, txt, 
            (int(tl[0]), int(tl[1]+scale*space*i)), 
            cv2.FONT_HERSHEY_COMPLEX, 
            scale, color, thickness)
    return img, i



if __name__ == '__main__':
    import fire
    fire.Fire(main)
