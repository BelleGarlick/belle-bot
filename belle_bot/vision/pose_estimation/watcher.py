import base64
import json

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
CLIENT.listen("vision/pose-estimation", get_bounding_boxes)


if __name__ == "__main__":
    cv2.namedWindow("preview")
    print("Object Detection Viewer Running")

    try:
        while True:

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
                shape = json.loads(predictions["shape"])
                predictions = np.frombuffer(predictions_bytes, np.float32)
                predictions = predictions.reshape(shape)

                print(predictions.shape)

                humans = shape[0]
                for human in range(humans):

                    def draw_line(idx1, idx2):
                        return cv2.line(
                            frame,
                            (int(predictions[human][idx1][0]), int(predictions[human][idx1][1])),
                            (int(predictions[human][idx2][0]), int(predictions[human][idx2][1])),
                            (0, 255, 0),
                            thickness=1,
                            lineType=8
                        )

                    frame = draw_line(0, 1)
                    frame = draw_line(1, 2)
                    frame = draw_line(0, 2)
                    frame = draw_line(2, 4)
                    frame = draw_line(1, 3)
                    frame = draw_line(0, 5)
                    frame = draw_line(0, 6)
                    frame = draw_line(5, 7)
                    frame = draw_line(6, 8)

                    for kp in list(predictions[human]):
                        frame = cv2.circle(
                            frame,
                            (int(kp[0]), int(kp[1])),
                            5,
                            (0, 255, 0),
                            -1
                        )

                # render the image
                cv2.imshow("preview", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
