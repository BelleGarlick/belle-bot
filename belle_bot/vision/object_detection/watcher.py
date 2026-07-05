import base64

import cv2
import queue

import numpy as np

from belle_bot.sensors import cameras
from belle_bot.fabric import FabricClient
from belle_bot.sensors.cameras.utils import parse_camera_stream

frame_queue = queue.Queue()
bounding_boxes_queue = queue.Queue()
CLIENT = FabricClient()


def show_frame_callback(data):
    frame_queue.put(data)


def get_bounding_boxes(boxes):
    bounding_boxes_queue.put(boxes)


CLIENT.listen(cameras.config.FABRIC_ID, show_frame_callback)
CLIENT.listen("vision/bounding_boxes", get_bounding_boxes)


if __name__ == "__main__":
    cv2.namedWindow("preview")

    try:
        while True:
            print("Object Detection Viewer Running")

            # Check if there is a new frame in the queue
            if not frame_queue.empty() and not bounding_boxes_queue.empty():
                # Only process the latest frame and drop everything else in the queue
                while not frame_queue.empty():
                    frame = frame_queue.get()

                while not bounding_boxes_queue.empty():
                    predictions = bounding_boxes_queue.get()

                # process the frame
                frame = parse_camera_stream(frame["rgb"])

                # process the predictions
                predictions_bytes = base64.b64decode(predictions["predictions"])
                predictions = np.frombuffer(predictions_bytes, np.float32)
                predictions = predictions.reshape((predictions.shape[0] // 6, 6))

                # Draw the predictions on the image
                for prediction in predictions:
                    # Only render hoomans
                    if prediction[-1] != 0:
                        continue

                    frame = cv2.rectangle(
                        frame,
                        (int(prediction[0]), int(prediction[1])),
                        (int(prediction[2]), int(prediction[3])),
                        (0, 255, 0),
                        2
                    )

                # render the image
                cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
