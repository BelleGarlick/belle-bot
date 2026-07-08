import base64

import cv2
import numpy as np


def parse_camera_stream(frame_bytes: str):
    frame_bytes = base64.b64decode(frame_bytes)
    return cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)


def parse_depth_stream(depth_bytes_str: str, shape=(480, 640), max_distance_mm=4000):
    """
    Decodes a base64-encoded 16-bit depth stream and converts it to an
    8-bit grayscale image.
    
    Args:
        depth_bytes_str: Base64-encoded string of the raw 16-bit depth frame.
        shape: Resolution tuple (height, width).
        max_distance_mm: Maximum depth distance in millimeters to map to white (255).
                         Values beyond this are clipped.
    """
    # Decode the base64 string to raw bytes
    depth_bytes = base64.b64decode(depth_bytes_str)
    
    # Convert from buffer to 16-bit unsigned integers and reshape
    depth_16 = np.frombuffer(depth_bytes, dtype=np.uint16).reshape(shape)
    
    # Clip values to the maximum distance and scale to 8-bit grayscale range (0-255)
    clipped = np.clip(depth_16, 0, max_distance_mm)
    depth_8 = (clipped * (255.0 / max_distance_mm)).astype(np.uint8)
    
    return depth_8
