import cv2
import queue

from belle_bot.sensors.cameras.config import FABRIC_ID
from belle_bot.fabric import FabricClient
from belle_bot.sensors.cameras.utils import parse_camera_stream

frame_queue = queue.Queue()
CLIENT = FabricClient()


def show_frame_callback(data):
    # Just put the data in the queue and return immediately
    frame_queue.put(data)


if __name__ == "__main__":
    CLIENT.listen(FABRIC_ID, show_frame_callback)

    cv2.namedWindow("preview")

    # Main loop: Must be on the main thread
    try:
        while True:
            # Check if there is a new frame in the queue
            if not frame_queue.empty():
                # Only process the latest frame and drop everything else in the queue
                while not frame_queue.empty():
                    data = frame_queue.get()

                frame = parse_camera_stream(data["rgb"])
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imshow("preview", frame)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()