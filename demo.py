'''XMem + Detic + Deep-Sort

usage: python demo.py ../R0-P03_03.mp4 --vocab '["peanut butter jar", "jelly jar", "plate", "cutting board", "paper towel", "tortilla, white circular flat", "burrito", "toothpicks", "floss", "person", "knife"]' --fps-down 3
'''
import sys
import time
import logging
from collections import defaultdict, Counter
from tqdm.contrib.logging import logging_redirect_tqdm

import cv2
import tqdm
import numpy as np
import torch
import supervision as sv

from xmem.inference import XMem, log
from xmem.util.box_annotator import BoxAnnotator
from torchvision.ops import masks_to_boxes

from detic import Detic


device = 'cuda'

# default configuration
xmem_config = {
    'top_k': 15,
    'mem_every': 30,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 64,
    'min_mid_term_frames': 6,
    'max_mid_term_frames': 12,
    'max_long_term_elements': 5000,
    # 'dilate_size_threshold': 0.01,
    'tentative_frames': 3,
    'max_age': 60 * 30,
}




import ipdb
@ipdb.iex
@torch.no_grad()
def mainx(*a, **kw):
    return main(*a, **kw)

def main(src, vocab='lvis', untracked_vocab=None, out_path='xmem_output.mp4', detect_every=1, skip_frames=0, fps_down=1, size=480, limit=None):
    if untracked_vocab:
        vocab = vocab + untracked_vocab
    
    # object detector
    detic_model = Detic(vocab, conf_threshold=0.5, masks=True).to(device)
    print(detic_model.labels)

    # object tracker
    xmem = XMem(xmem_config).eval().to(device)

    # object label counter (dict of object_track_id -> {plate: 25, cutting_board: 1})
    # this should have the same keys as ds_tracker
    label_counts = defaultdict(lambda: Counter())

    draw = Drawer()

    # video read-write loop
    video_info, WH = get_video_info(src, size, fps_down, ncols=2)
    afps = video_info.fps / fps_down
    i_detect = int(detect_every*video_info.og_fps//fps_down)
    print('detecting every', i_detect, detect_every, afps, detect_every%(1/afps))
    
    with sv.VideoSink(target_path=out_path, video_info=video_info) as s:
        for i, frame in enumerate(tqdm.tqdm(sv.get_video_frames_generator(src))):
            if i < skip_frames or i%fps_down: continue
            if limit and i > limit: break

            frame = cv2.resize(frame, WH)

            # run detic
            detections = det_mask = None
            if not i % i_detect or i == skip_frames:
                # get object detections
                outputs = detic_model(frame)
                det_mask = outputs["instances"].pred_masks.int()

                # draw detic
                detect_out_frame, detections = draw.draw_detectron2(frame, outputs, detic_model.labels)
                log.info(f'Detected ({i}|{i_detect}): {detic_model.labels[detections.class_id]}')

                if untracked_vocab:
                    keep = [l not in untracked_vocab for l in detic_model.labels[detections.class_id]]
                    det_mask = det_mask[keep]
                    detections = detections[keep]
                
                det_mask = xmem.dilate_masks(det_mask)

            # run xmem
            pred_mask, track_ids, input_track_ids = xmem(frame, det_mask, only_confirmed=True)

            # update label counts
            if detections is not None:
                for (ti, l) in zip(input_track_ids, detections.class_id):
                    label_counts[ti].update([l])

            # draw xmem
            track_out_frame, detections = draw.draw_tracks(frame, pred_mask, label_counts, track_ids, detic_model.labels)

            # write frame to file
            s.write_frame(np.concatenate([track_out_frame, detect_out_frame], axis=1))

    print("wrote to:", out_path)




class Drawer:
    def __init__(self):
        self.box_ann = BoxAnnotator()
        self.mask_ann = sv.MaskAnnotator()

    def draw_detectron2(self, frame, outputs, labels):
        detections, det_labels = self.prepare_detectron2_detections(outputs, labels)
        frame = self.draw_detectron2_detections(frame, detections, det_labels)
        return frame, detections
    
    def draw_tracks(self, frame, pred_mask, label_counts, track_ids, labels):
        track_detections, det_labels = self.prepare_track_detections(pred_mask, label_counts, track_ids, labels)
        frame = self.draw_track_detections(frame, track_detections, det_labels)
        return frame, track_detections

    def prepare_detectron2_detections(self, outputs, labels):
        detections = sv.Detections(
            xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
            mask=outputs["instances"].pred_masks.int().cpu().numpy(),
            confidence=outputs["instances"].scores.cpu().numpy(),
            class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int),
        )
        return detections, labels

    def draw_detectron2_detections(self, frame, detections, labels):
        # draw detic detections
        frame = frame.copy()
        frame = self.mask_ann.annotate(frame, detections)
        frame = self.box_ann.annotate(
            frame, detections, 
            labels=labels[detections.class_id]
        )
        return frame
    
    def prepare_track_detections(self, pred_mask, label_counts, track_ids, labels):
        # convert to Detection object for visualization
        track_detections = sv.Detections(
            mask=pred_mask.cpu().numpy(),
            xyxy=masks_to_boxes(pred_mask).cpu().numpy(),
            class_id=np.array([label_counts[i].most_common(1)[0][0] for i in track_ids]),
            tracker_id=track_ids,
        )

        # draw xmem detections
        labels = [
            [f'{i} {labels[l][:12]} {c}' 
                for l, c in label_counts[i].most_common(2)] 
            for i in track_detections.tracker_id
        ]
        return track_detections, labels
    
    def draw_track_detections(self, frame, track_detections, labels):
        frame = frame.copy()
        track_detections.class_id = track_detections.tracker_id  # for color
        frame = self.mask_ann.annotate(frame, track_detections)
        frame = self.box_ann.annotate(
            frame, track_detections,
            labels=labels
        )
        return frame


def get_video_info(src, size, fps_down=1, nrows=1, ncols=1):
    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
    aspect = video_info.width / video_info.height
    video_info.width = int(aspect*size)//16*16
    video_info.height = int(size)//16*16
    WH = video_info.width, video_info.height

    # double width because we have both detic and xmem frames
    video_info.width *= ncols
    video_info.height *= nrows
    # possibly reduce the video frame rate
    video_info.og_fps = video_info.fps
    video_info.fps /= fps_down or 1

    print(f"Input Video {src}\nsize: {WH}  fps: {video_info.fps}")
    return video_info, WH

def main_profile(*a, **kw):
    # from xmem.model.network import XMem as XMemModel
    # from xmem.model.modules import Decoder
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp.add_function(XMem.forward)
    # lp.add_function(Decoder.forward)
    # lp.add_function(main)
    # try:
    #     lp(mainx)(*a, **kw)
    # finally:
    #     lp.print_stats()

    from pyinstrument import Profiler
    prof = Profiler(async_mode='disabled')
    try:
        with prof:
            mainx(*a, **kw)
    finally:
        prof.print()

    # import torch.profiler
    # prof = torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profile'),
    #     schedule=torch.profiler.schedule(skip_first=10, wait=60, warmup=10, active=20, repeat=5),
    #     record_shapes=True)
    # try:
    #     mainx(*a, **kw)
    # finally:
    #     print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))

if __name__ == '__main__':
    import fire
    logging.basicConfig(level=logging.DEBUG)
    with logging_redirect_tqdm():
        fire.Fire(main_profile)
