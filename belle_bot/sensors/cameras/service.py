import base64
import json
import time
import uuid

import cv2

from belle_bot.fabric import FabricClient
from belle_bot.sensors.cameras import config

CLIENT = FabricClient()

if __name__ == "__main__":
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    target_interval = 1.0 / config.MAX_FPS
    while rval:
        start_time = time.perf_counter()

        frame = cv2.resize(frame, (920, 512))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
        frame_string = base64.b64encode(buffer).decode("utf-8")

        CLIENT.publish(config.FABRIC_ID, {
            "frame_id": str(uuid.uuid4()),
            "rgb": frame_string,
            "shape": json.dumps(frame.shape),
            "jpeg_quality": config.JPEG_QUALITY,
        })

        rval, frame = vc.read()

        elapsed_time = time.perf_counter() - start_time
        sleep_time = max(0.0, target_interval - elapsed_time)
        time.sleep(sleep_time)

    vc.release()
