import cv2
import numpy as np
from ultralytics.utils import nms


CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


COLOR_PALETTE = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)


def perform_nms(
    raw_predictions,
    conf_threshold=0.3,
    iou_threshold=0.6,
):
    return nms.non_max_suppression(
        raw_predictions,
        conf_thres=conf_threshold,
        iou_thres=iou_threshold,  # allow callers (e.g. TrackTrack loose-NMS recovery) to override IoU
        classes=None,
        agnostic=False,
        max_det=300,
        nc=0,
        end2end=True,
        rotated=False,
        return_idxs=False,
    )


def render_boxes(img, boxes):
    img = img.copy()

    def clip(v, min_v, max_v):
        return max(min(v, max_v), min_v)

    for x_start, y_start, x_end, y_end, confidence, class_id in boxes:
        # Convert to integer pixel values ONLY here for CV2 drawing functions
        x_start = clip(int(round(x_start)), 0, img.shape[1] - 1)
        y_start = clip(int(round(y_start)), 0, img.shape[1] - 1)
        x_end = clip(int(round(x_end)), 0, img.shape[1] - 1)
        y_end = clip(int(round(y_end)), 0, img.shape[1] - 1)

        x_end = max(y_start+1, x_end)
        y_end = max(y_start+1, y_end)

        # Defensive check
        if (x_end - x_start) <= 0 or (y_end - y_start) <= 0:
            continue

        color = get_color(int(class_id))
        img = cv2.rectangle(img, (x_start, y_start), (x_end, y_end), color, 2)
        img = cv2.putText(
            img,
            f"{CLASSES[min(len(CLASSES) - 1, max(0, int(class_id)))]}: {confidence:.2f}",
            (x_start, max(y_start - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return img


def get_color(class_id):
    """Safely retrieves BGR color using modulo indexing."""
    return [int(c) for c in COLOR_PALETTE[int(class_id) % len(COLOR_PALETTE)]]
