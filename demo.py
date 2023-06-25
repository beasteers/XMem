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

from xmem.inference import XMem
from xmem.inference.interact.interactive_utils import image_to_torch
from xmem.util.box_annotator import BoxAnnotator
from torchvision.ops import masks_to_boxes

from detic import Detic


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
    'dilate_size_threshold': 0.01,
}


@torch.no_grad()
def main(src, vocab='lvis', detect_every=1, skip_frames=0, fps_down=1, size=480):
    # object detector
    detic_model = Detic(vocab, conf_threshold=0.5, masks=True).to(device)
    print(detic_model.labels)

    # object tracker
    xmem = XMem(xmem_config).eval().to(device)

    # object label counter (dict of object_track_id -> {plate: 25, cutting_board: 1})
    # this should have the same keys as ds_tracker
    label_counts = defaultdict(lambda: Counter())

    video_info, WH = get_video_info(src, size, fps_down, ncols=2)

    box_ann = BoxAnnotator()
    mask_ann = sv.MaskAnnotator()

    with sv.VideoSink(target_path='xmem_output.mp4', video_info=video_info) as s:
        for i, frame in enumerate(tqdm.tqdm(sv.get_video_frames_generator(src))):
            if i < skip_frames or i%fps_down: continue

            frame = cv2.resize(frame, WH)

            detections = mask = class_ids = None
            if not i % int(detect_every*video_info.og_fps) or i == skip_frames:
                # get object detections
                outputs = detic_model(frame)
                mask = outputs["instances"].pred_masks.int()
                detections = sv.Detections(
                    xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
                    mask=mask.cpu().numpy(),
                    confidence=outputs["instances"].scores.cpu().numpy(),
                    class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int),
                )
                class_ids = detections.class_id

                # draw detic detections
                detect_out_frame = frame.copy()
                detect_out_frame = mask_ann.annotate(detect_out_frame, detections)
                detect_out_frame = box_ann.annotate(
                    detect_out_frame, detections, 
                    labels=detic_model.labels[detections.class_id])

            # run xmem
            X, _ = image_to_torch(frame, device=device)
            pred_mask, track_ids, input_track_ids = xmem(X, mask)

            # update label counts
            if class_ids is not None:
                for (ti, l) in zip(input_track_ids, class_ids):
                    label_counts[ti].update((l,))

            # convert to Detection object for visualization
            track_detections = sv.Detections(
                mask=pred_mask.cpu().numpy(),
                xyxy=masks_to_boxes(pred_mask).cpu().numpy(),
                class_id=np.array([label_counts[i].most_common(1)[0][0] for i in track_ids]),
                tracker_id=track_ids,
            )

            # draw xmem detections
            labels = [
                [f'{i} {detic_model.labels[l][:12]} {c}' 
                 for l, c in label_counts[i].most_common(2)] 
                for i in track_detections.tracker_id
            ]
            track_out_frame = frame.copy()
            track_detections.class_id = track_detections.tracker_id  # for color
            track_out_frame = mask_ann.annotate(track_out_frame, track_detections)
            track_out_frame = box_ann.annotate(
                track_out_frame, track_detections,
                labels=labels)

            # write frame to file
            s.write_frame(np.concatenate([track_out_frame, detect_out_frame], axis=1))


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
    return video_info, WH


if __name__ == '__main__':
    import fire
    fire.Fire(main)
