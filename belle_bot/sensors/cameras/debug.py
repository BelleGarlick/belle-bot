import base64
import json
import time
import uuid

import cv2

from belle_bot.infra.fabric import FabricClient

CLIENT = FabricClient()
FABRIC_ID = "sensors/camera"
MAX_FPS = 10
JPEG_QUALITY = 60

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)

    target_interval = 1.0 / MAX_FPS

    while True:
        start_time = time.perf_counter()
        ret, frame = cam.read()

        color_image = frame

        if color_image is None:
            continue

        # Convert both from RGB (RealSense) to BGR (OpenCV)
        color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        success, color_buffer = cv2.imencode('.jpg', color_bgr, encode_param)

        color_frame_string = base64.b64encode(color_buffer).decode("utf-8")

        CLIENT.publish(FABRIC_ID, {
            "frame_id": str(uuid.uuid4()),
            "rgb": color_frame_string,
            "shape": json.dumps(color_image.shape),
            "jpeg_quality": JPEG_QUALITY,
        })

        # Rate limit the publishing rate to match config.MAX_FPS
        elapsed_time = time.perf_counter() - start_time
        sleep_time = max(0.0, target_interval - elapsed_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
