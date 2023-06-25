import itertools

import cv2
import numpy as np
import supervision as sv



# bounding box drawing utils
# supervision's builtin one blocks too much of the video

class BoxAnnotator(sv.BoxAnnotator):
    def annotate(self, image, detections, labels=None):
        return draw_boxes(image, detections.xyxy, labels)


def draw_boxes(im, boxes, labels=None, color=(0,255,0), size=1, text_color=(0, 0, 255), spacing=3):
    boxes = np.asarray(boxes).astype(int)
    color = np.asarray(color).astype(int)
    color = color[None] if color.ndim == 1 else color
    labels = itertools.chain([] if labels is None else labels, itertools.cycle(['']))
    for xy, c in zip(boxes, itertools.cycle(color)):
        im = cv2.rectangle(im, xy[:2], xy[2:4], tuple(c.tolist()), 2)
    
    for xy, label, c in zip(boxes, labels, itertools.cycle(color)):
        if label:
            if isinstance(label, list):
                im, _ = draw_text_list(
                    im, label, 0, tl=xy[:2] + spacing, scale=im.shape[1]/1400*size, 
                    space=40, color=text_color)
            else:
                im = cv2.putText(
                    im, label, xy[:2] - spacing, 
                    cv2.FONT_HERSHEY_SIMPLEX, im.shape[1]/1400*size, 
                    text_color, 1)
    return im


def draw_text_list(img, texts, i=-1, tl=(10, 50), scale=0.4, space=50, color=(255, 255, 255), thickness=1):
    for i, txt in enumerate(texts, i+1):
        cv2.putText(
            img, txt, 
            (int(tl[0]), int(tl[1]+scale*space*i)), 
            cv2.FONT_HERSHEY_COMPLEX , 
            scale, color, thickness)
    return img, i

