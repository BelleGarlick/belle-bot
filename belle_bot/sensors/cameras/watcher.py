import cv2
import queue

from belle_bot.sensors.cameras.config import FABRIC_ID
from belle_bot.fabric import FabricClient
from belle_bot.sensors.cameras.utils import parse_camera_stream, parse_depth_stream

frame_queue = queue.Queue()
CLIENT = FabricClient()


def show_frame_callback(data):
    # Just put the data in the queue and return immediately
    frame_queue.put(data)


if __name__ == "__main__":
    CLIENT.listen(FABRIC_ID, show_frame_callback)

    cv2.namedWindow("preview")

    try:
        while True:
            # Block until at least one frame is received (saves CPU)
            data = frame_queue.get()

            # Flush any older frames that accumulated while we were processing/rendering
            while not frame_queue.empty():
                try:
                    data = frame_queue.get_nowait()
                except queue.Empty:
                    break

            frame = parse_depth_stream(data["depth"])
            print(frame.shape)
            cv2.imshow("preview", frame)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()