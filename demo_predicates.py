'''XMem + Detic + Deep-Sort

usage: python demo.py ../R0-P03_03.mp4 --vocab '["peanut butter jar", "jelly jar", "plate", "cutting board", "paper towel", "tortilla, white circular flat", "burrito", "toothpicks", "floss", "person", "knife"]' --fps-down 3
'''
import time
from collections import defaultdict, Counter
import cv2
import tqdm
from PIL import Image
import numpy as np
import torch
import supervision as sv

from xmem.inference import XMem, Track
from xmem.inference.interact.interactive_utils import image_to_torch
from xmem.util.box_annotator import BoxAnnotator
from torchvision.ops import masks_to_boxes

from detic import Detic
from egohos import EgoHos
import clip


device = 'cuda'

class CustomTrack(Track):
    hoi_class_id = 0
    state_class_label = ''
    def __init__(self, track_id, t_obs, n_init=3, **kw):
        super().__init__(track_id, t_obs, n_init, **kw)
        self.label_count = Counter()
        self.z_clips = {}


class FewShot(torch.nn.Module):
    def __init__(self, classifiers):
        super().__init__()
        # image encoder
        self.model, self.pre = clip.load("ViT-B/32", device=device)
        # classifier
        self.classifiers = {
            k: [torch.as_tensor(d['Z']).to(device), d['labels']]
            for k, d in classifiers.items()
        }

    def can_classify(self, labels):
        return np.asarray([l in self.classifiers for l in labels])

    def encode_boxes(self, img, boxes):
        # encode each bounding box crop with clip
        Z = self.model.encode_image(torch.stack([
            self.pre(Image.fromarray(img[
                int(y):max(int(np.ceil(y2)), int(y+2)),
                int(x):max(int(np.ceil(x2)), int(x+2))]))
            for x, y, x2, y2 in boxes.cpu()
        ]).to(device))
        Z /= Z.norm(dim=1, keepdim=True)
        return Z
    
    def classify(self, Z, labels):
        outputs = []
        for z, l in zip(Z, labels):
            z_cls, txt_cls = self.classifiers[l]
            out = (z @ z_cls.t()).softmax(dim=-1).cpu().numpy()
            i = np.argmax(out)
            outputs.append(txt_cls[i])
        return np.atleast_1d(np.array(outputs))

    def forward(self, img, boxes, labels):
        valid = self.can_classify(labels)
        if not valid.any():
            return np.array([None]*len(boxes))
        labels = np.asanyarray(labels)
        Z = self.encode_boxes(img, boxes[valid])
        clses = self.classify(Z, labels[valid])
        all_clses = np.array([None]*len(boxes))
        all_clses[valid] = clses
        return all_clses


# default configuration
xmem_config = {
    'top_k': 20,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 64,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
    'dilate_size_threshold': 0, #.01,
    'tentative_frames': 2,
    'tentative_age': 3,
    'min_iou': 0.2,
}


@torch.no_grad()
def main(src, vocab='lvis', untracked_vocab=None, out_path='xmem_predc_output.mp4', detect_every=0.2, skip_frames=0, fps_down=1, size=480):

    TORTILLA = 'tortilla, white flour, circular'
    # PB_JAR = 'peanut butter jar'
    # JELLY_JAR = 'jelly jar'
    # vocab = [
    #     TORTILLA, 
    #     # PB_JAR, JELLY_JAR, 
    #     # 'peanut butter', 'jelly', 
    #     # 'paper towel', 'cutting board',
    #     # 'person', 'plate',
    #     # 'knife'
    # ]
    if untracked_vocab:
        vocab = vocab + untracked_vocab

    # object detector
    detic = Detic(vocab, conf_threshold=0.7, masks=True).to(device)
    vocab = detic.labels
    print(vocab)

    # hand-object interactions
    egohos = EgoHos(mode='obj1', device=device)
    egohos_classes = np.array(list(egohos.CLASSES))

    # few-shot classifier
    fewshot = FewShot({
        TORTILLA: np.load('tortilla_pinwheels.npz'),
    })

    # object tracker
    xmem = XMem(xmem_config, Track=CustomTrack).eval().to(device)

    # box_ann = sv.BoxAnnotator(text_scale=0.4, text_padding=1)
    # mask_ann = sv.MaskAnnotator()
    draw = Drawer()

    detect_out_frame = hos_out_frame = None
    video_info, WH, WH2 = get_video_info(src, size, fps_down, nrows=2, ncols=2)
    afps = video_info.fps / fps_down
    i_detect = 1 if detect_every is True else int(detect_every*video_info.og_fps//fps_down)
    print('detecting every', i_detect, detect_every, afps, detect_every%(1/afps))

    with sv.VideoSink(target_path=out_path, video_info=video_info) as s:
        for i, frame in enumerate(tqdm.tqdm(sv.get_video_frames_generator(src))):
            if i < skip_frames or i%fps_down: continue

            frame = cv2.resize(frame, WH)

            detections = det_mask = None
            hoi_masks = hoi_class_ids = hand_masks = None

            # get hoi detections
            hoi_masks, hoi_class_ids = egohos(frame)
            hos_out_frame, hos_detections = draw.draw_egohos(frame, hoi_masks, hoi_class_ids, egohos)
            hand_masks = hoi_masks[(hoi_class_ids == 1) | (hoi_class_ids == 2)].sum(0)

            if not i % i_detect or detect_out_frame is None:
                # get object detections
                outputs = detic(frame)
                nms_indices = asymmetric_nms(
                    outputs['instances'].pred_boxes.tensor.cpu().numpy(), 
                    outputs['instances'].scores.cpu().numpy())
                outputs['instances'] = outputs['instances'][nms_indices]
                det_mask = outputs["instances"].pred_masks.int()

                detect_out_frame, detections = draw.draw_detectron2(frame, outputs, detic.labels)

                # filter out objects we want to detect but not track (for disambiguation)
                if untracked_vocab:
                    keep = np.array([l not in untracked_vocab for l in detic.labels[detections.class_id]], dtype=bool)
                    det_mask = det_mask[keep]
                    detections = detections[keep]

                tqdm.tqdm.write(f'{len(detections)} {len(hos_detections)} {len(xmem.track_ids)}')

            # run xmem
            pred_mask, track_ids, input_track_ids = xmem(frame, det_mask, negative_mask=hand_masks)
            pred_boxes = masks_to_boxes(pred_mask)

            # update label counts
            if detections is not None:
                for (ti, i) in zip(input_track_ids, detections.class_id):
                    xmem.tracks[ti].label_count.update([i])

            # update HOI
            if hoi_class_ids is not None:
                hoi_track_ids, hoi_idxs = xmem.match_iou(pred_mask, hoi_masks)
                hoi_track_ids = track_ids[hoi_track_ids]
                for (ti, i) in zip(hoi_track_ids, hoi_idxs):
                    xmem.tracks[ti].hoi_class_id = hoi_class_ids[i]
                for ti in set(track_ids) - set(hoi_track_ids):
                    xmem.tracks[ti].hoi_class_id = 0

            # get track labels
            pred_class_ids = np.array([xmem.tracks[i].label_count.most_common(1)[0][0] for i in track_ids]).astype(int)
            pred_labels = vocab[pred_class_ids]

            # update CLIP classification
            cls_outputs = fewshot(frame, pred_boxes, pred_labels)
            for ti, cls_label in zip(track_ids, cls_outputs):
                xmem.tracks[ti].state_class_label = cls_label or ''

            # convert to Detection object for visualization
            display_subset = (pred_labels == TORTILLA)# & visible_tracks
            track_out_frame, track_detections = draw.draw_tracks(
                frame, 
                pred_mask[display_subset], 
                pred_boxes[display_subset], 
                track_ids[display_subset], 
                xmem, detic, egohos)

            # draw everything together - (missing untracked vocab)
            full_out_frame, full_detections = draw.draw_tracks(
                frame, 
                pred_mask, 
                pred_boxes, 
                track_ids, 
                xmem, detic, egohos)

            hand_masks=np.stack([hand_masks.cpu().numpy()*255]*3, axis=-1).astype(np.uint8)
            # write frame to file
            x = np.concatenate([
                np.concatenate([hand_masks, track_out_frame], axis=1),
                np.concatenate([detect_out_frame, hos_out_frame], axis=1),
            ], axis=0)
            s.write_frame(x)



def asymmetric_nms(boxes, scores, iou_threshold=0.7):
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Sort boxes by their confidence scores in descending order
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    selected_indices = []
    while len(boxes) > 0:
        # Pick the box with the highest confidence score
        b = boxes[0]
        selected_indices.append(indices[0])

        # Calculate IoU between the picked box and the remaining boxes
        intersection_area = (
            np.maximum(0, np.minimum(b[2], boxes[1:, 2]) - np.maximum(b[0], boxes[1:, 0])) * 
            np.maximum(0, np.minimum(b[3], boxes[1:, 3]) - np.maximum(b[1], boxes[1:, 1]))
        )
        smaller_box_area = np.minimum(areas[0], areas[1:])
        iou = intersection_area / (smaller_box_area + 1e-7)

        # Filter out boxes with IoU above the threshold
        filtered_indices = np.where(iou <= iou_threshold)[0]
        indices = indices[filtered_indices + 1]
        boxes = boxes[filtered_indices + 1]
        areas = areas[filtered_indices + 1]

    return selected_indices


class Drawer:
    def __init__(self):
        self.box_ann = sv.BoxAnnotator(text_scale=0.4, text_padding=1)
        self.mask_ann = sv.MaskAnnotator()

    def draw_detectron2(self, frame, outputs, labels):
        detections, det_labels = self.prepare_detectron2_detections(outputs, labels)
        frame = self.draw_detections(frame, detections, det_labels)
        return frame, detections
    
    def draw_egohos(self, frame, hoi_masks, hoi_class_ids, egohos):
        detections, det_labels = self.prepare_egohos_detections(hoi_masks, hoi_class_ids, egohos)
        frame = self.draw_detections(frame, detections, det_labels)
        return frame, detections
    
    def draw_tracks(self, frame, pred_mask, pred_boxes, track_ids, xmem, detic, egohos):
        track_detections, det_labels = self.prepare_track_detections(pred_mask, pred_boxes, track_ids, xmem, detic, egohos)
        frame = self.draw_detections(frame, track_detections, det_labels)
        return frame, track_detections

    def draw_detections(self, frame, detections, labels):
        frame = frame.copy()
        frame = self.mask_ann.annotate(frame, detections)
        frame = self.box_ann.annotate(frame, detections, labels=labels)
        return frame
    
    def prepare_detectron2_detections(self, outputs, labels):
        detections = sv.Detections(
            xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
            mask=outputs["instances"].pred_masks.int().cpu().numpy(),
            confidence=outputs["instances"].scores.cpu().numpy(),
            class_id=outputs["instances"].pred_classes.cpu().numpy().astype(int),
        )
        labels = labels[detections.class_id]
        return detections, labels
    
    def prepare_egohos_detections(self, hoi_masks, hoi_class_ids, egohos):
        detections = sv.Detections(
            xyxy=masks_to_boxes(hoi_masks).cpu().numpy(),
            mask=hoi_masks.cpu().numpy(),
            class_id=hoi_class_ids,
        )
        labels = egohos.CLASSES[detections.class_id]
        return detections, labels

    def prepare_track_detections(self, pred_mask, pred_boxes, track_ids, xmem, detic, egohos):
        # convert to Detection object for visualization
        track_detections = sv.Detections(
            mask=pred_mask.cpu().numpy(),
            xyxy=pred_boxes.cpu().numpy(),
            # class_id=np.array([xmem.tracks[i].label_count.most_common(1)[0][0] for i in track_ids]),
            # class_id=np.array([xmem.tracks[i].hoi_class_id for i in track_ids]),
            class_id=track_ids,
            tracker_id=track_ids,
        )

        # draw xmem detections
        labels = [
            [f'{detic.labels[l].split(",")[0]}' 
                for l, c in xmem.tracks[i].label_count.most_common(1)] + 
            [
                egohos.CLASSES[xmem.tracks[i].hoi_class_id],
                xmem.tracks[i].state_class_label,
            ]
            for i in track_detections.tracker_id
        ]
        labels = [' | '.join([l for l in ls if l]) for ls in labels]
        return track_detections, labels



def get_video_info(src, size, fps_down=1, nrows=1, ncols=1, render_scale=1):
    # get the source video info
    video_info = sv.VideoInfo.from_video_path(video_path=src)
    # make the video size a multiple of 16 (because otherwise it won't generate masks of the right size)
    aspect = video_info.width / video_info.height
    video_info.width = int(aspect*size)//16*16
    video_info.height = int(size)//16*16
    WH = video_info.width, video_info.height
    video_info.width *= render_scale
    video_info.height *= render_scale
    WH2 = video_info.width, video_info.height

    # double width because we have both detic and xmem frames
    video_info.width *= ncols
    video_info.height *= nrows
    # possibly reduce the video frame rate
    video_info.og_fps = video_info.fps
    video_info.fps /= fps_down or 1

    print(f"Input Video {src}\nsize: {WH} {WH2}  fps: {video_info.fps}")
    return video_info, WH, WH2


if __name__ == '__main__':
    import fire
    fire.Fire(main)
