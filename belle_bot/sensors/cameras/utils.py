import base64

import cv2
import numpy as np


def parse_camera_stream(frame_bytes: str):
    frame_bytes = base64.b64decode(frame_bytes)
    return cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
