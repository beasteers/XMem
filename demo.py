# python demo.py ../R0-P03_03.mp4 --vocab '["peanut butter jar", "jelly jar", "plate", "cutting board", "paper towel", "tortilla, white circular flat", "burrito", "toothpicks", "floss", "person", "knife"]' --fps-down 3
import time
from supervision import VideoInfo, VideoSink, get_video_frames_generator
import cv2
import tqdm
import numpy as np
import torch

from xmem.inference import InferenceCore, XMem
from xmem.inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
from torchvision.ops import masks_to_boxes
import deep_sort
from torch.profiler import profile, record_function, ProfilerActivity



device = 'cuda'

# default configuration
config = {
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

# WH = (720, 480)

from collections import defaultdict, Counter

object_labels = defaultdict(lambda: Counter())

# def run_profile(func):
#     def inner(*a, **kw):
#         from pyinstrument import Profiler
#         profiler = Profiler()
#         try:
#             profiler.start()
#             return func(*a, **kw)
#         finally:
#             profiler.stop()
#             profiler.print()
#     return inner
from contextlib import contextmanager
# @contextmanager
# def run_profile():
#     from pyinstrument import Profiler
#     profiler = Profiler()
#     try:
#         profiler.start()
#         yield
#     finally:
#         profiler.stop()
#         profiler.print()
@contextmanager
def run_profile():
    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True)
    try:
        with prof: yield prof
    finally:
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=10))
# def run_profile(func):
#     def inner(*a, **kw):
#         prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
#         try:
#             with prof:
#                 return func(*a, **kw)
#         finally:
#             print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#     return inner


# @run_profile
@torch.no_grad()
def main(src, vocab='lvis', detect_every=1, skip_frames=30, fps_down=1, size=480):
    from detic import Detic
    from ptgprocess.util import draw_boxes

    WH = (int(720/480*size)//16*16, int(size)//16*16)

    # object detector
    detic_model = Detic(['cat'], conf_threshold=0.5, masks=True).to(device)
    detic_model.set_vocab(vocab)
    print(detic_model.labels)

    # box tracker
    ds_tracker = deep_sort.Tracker(deep_sort.NearestNeighbor('cosine', 0.2), max_age=None)

    # mask tracker
    network = XMem(config).eval().to(device)
    processor = InferenceCore(network, config=config)

    kernel = np.ones((5, 5), dtype=np.int32)

    video_info = VideoInfo.from_video_path(video_path=src)
    video_info.width, video_info.height = WH
    video_info.width *= 2
    og_fps = video_info.fps
    video_info.fps /= fps_down
    t_obs = 0
    with VideoSink(target_path='xmem_output.mp4', video_info=video_info) as s:#, run_profile() as prof:
        for i, frame in enumerate(tqdm.tqdm(get_video_frames_generator(src))):
            if i < skip_frames:
                continue
            if i%fps_down: continue
            frame = cv2.resize(frame, WH)
            out_frame = frame.copy()
            if not i % 200:
                print(processor.memory)
            labels = mask = None
            if not i % int(detect_every*og_fps) or i == skip_frames:
                # with record_function("detic"):
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
                # costs, track_ids = ds_tracker.gated_metric(detections)
                # costs = dict(zip(track_ids, costs))
                # print(labels)
                # print(costs)
                # for i, c in costs.items():
                #     print(i, c.min(), c.max())
                # input()

            frame_torch, _ = image_to_torch(frame, device=device)
            # with record_function("xmem"):
            prediction, track_ids = processor.step(frame_torch, mask)
            all_labels = list(processor.all_labels)
            if labels is not None:
                if track_ids is None:
                    track_ids = processor.all_labels
                # update matches in deep sort                
                matches = [(ti, di) for di, ti in enumerate(track_ids)]
                deleted = ds_tracker._update(detections, t_obs, matches)
                deleted and print('deleting', set(deleted))
                for i in deleted:
                    processor.delete_object_id(i)
                    # print('deleted', i)
                # print('tracks', set(ds_tracker.tracks))

                for ((ti, di), l) in zip(matches, labels):
                    object_labels[ti].update((l,))
                #     print(i, j, l, object_labels[i].most_common(2))
                # input()

                draw_boxes(out_frame2, xywh, [
                    [f'{i} {l}'] 
                    for i, l in zip(track_ids, detic_model.labels[cls_ids])
                ])

            if len(prediction):
                pred_mask = torch_prob_to_numpy_mask(prediction)
                # print(np.unique(pred_mask))
                out_frame = overlay_davis(out_frame, pred_mask)
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

                # if mask is not None:
                #     ids=np.unique(pred_mask)[1:]
                #     pm=torch.zeros((max(processor.all_labels)+1, *pred_mask.shape))
                #     print(pm.shape, ids)
                #     for i in ids:
                #         pm[i-1, pred_mask == i] = 1
                #     for i, mi in enumerate(mask):
                #         cv2.imwrite(f'ims/match-{labels.tolist()[i]}-{object_ids[i]}-{i}.jpg', np.concatenate([
                #             pm[object_ids[i]].cpu().numpy()*255, 
                #             mi.cpu().numpy()*255, 
                #         ], 1))
                #     input()
            # if mask is not None:
            #     mask1=torch.argmax(torch.cat([torch.zeros(1, *mask.shape[1:]), mask.cpu()]), dim=0)
            #     out_frame = overlay_davis(out_frame, mask1)
            # cv2.imshow('xmem', out_frame)
            # cv2.waitKey(1)
            s.write_frame(np.concatenate([out_frame, out_frame2], axis=1))

if __name__ == '__main__':
    import fire
    fire.Fire(main)
