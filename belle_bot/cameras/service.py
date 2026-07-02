import base64
import json
import time
import cv2

from belle_bot.fabric import FabricClient

MAX_FPS = 5
JPEG_QUALITY = 60 #%
CLIENT = FabricClient()

if __name__ == "__main__":
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        frame = cv2.resize(frame, (920, 512))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        frame_string = base64.b64encode(buffer).decode("utf-8")

        CLIENT.publish("camera", {
            "rgb": frame_string,
            "shape": json.dumps(frame.shape),
        })

        rval, frame = vc.read()

        # todo need to sort this out properly since this wont always lead to 5s time
        time.sleep(MAX_FPS / 1000)

    vc.release()
