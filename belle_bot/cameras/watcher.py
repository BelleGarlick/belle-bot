import base64
import cv2
import numpy as np
import json
import queue

from belle_bot.fabric import FabricClient

# Thread-safe queue for UI data
frame_queue = queue.Queue()
CLIENT = FabricClient()


def show_frame_callback(data):
    # Just put the data in the queue and return immediately
    frame_queue.put(data)


if __name__ == "__main__":
    CLIENT.listen("camera", show_frame_callback)

    cv2.namedWindow("preview")

    # Main loop: Must be on the main thread
    try:
        while True:
            # Check if there is a new frame in the queue
            if not frame_queue.empty():
                data = frame_queue.get()

                frame_bytes = base64.b64decode(data["rgb"])
                # Decode from JPEG format
                frame_array = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
                cv2.imshow("preview", frame_array)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()