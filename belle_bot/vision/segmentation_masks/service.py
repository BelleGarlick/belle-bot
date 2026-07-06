import asyncio
import json

import cv2
import numpy as np

from ultralytics import YOLO

from belle_bot.sensors import cameras
from belle_bot.fabric import FabricClient
from belle_bot.sensors.cameras.utils import parse_camera_stream
from belle_bot.vision.utils import letterbox_image

frame_queue = None
CLIENT = FabricClient()
loop = None


model = YOLO("yolo26n-seg.pt")

# todo make this testable

def show_frame_callback(data):
    loop.call_soon_threadsafe(frame_queue.put_nowait, data)


def rescale_frame(frame, new_size=640):
    frame_shape = frame.shape
    rescale_ratio = frame_shape[1] / new_size
    new_height = int(frame_shape[0] / rescale_ratio)
    return cv2.resize(frame, (new_size, new_height)), rescale_ratio


def reframe_image(frame):
    vertical_padding_offset = (640 - frame.shape[0]) // 2
    vertical_padding = np.zeros((vertical_padding_offset, 640, 3), dtype=np.int8)
    return np.concatenate([vertical_padding, frame, vertical_padding], axis=0).astype(np.uint8), vertical_padding_offset


def predict_bounding_boxes(frame, confidence_threshold=0.2):
    # r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    # frame = np.concatenate([np.expand_dims(x, axis=0) for x in (r, g, b)], axis=0)

    # frame = np.expand_dims(frame / 255, axis=0).astype(np.float32)

    # predictions = ort_sess.run(None, {"images": frame})[0][0]

    # todo switch this all out eventually when using onnx
    results = model(frame)

    return results[0]


async def main():
    global frame_queue, loop
    frame_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    CLIENT.listen(cameras.config.FABRIC_ID, show_frame_callback)

    while True:
        print("Segmentation Running")

        # Check if there is a new frame in the queue
        # In async, we wait for at least one item
        data = await frame_queue.get()

        # Only process the latest frame and drop everything else in the queue
        while not frame_queue.empty():
            data = frame_queue.get_nowait()

        frame = parse_camera_stream(data["rgb"])

        # Apply padding so it's square for the image
        frame, scale, vertical_padding = letterbox_image(frame, size=640)

        # Run predictions
        predictions = predict_bounding_boxes(frame)

        masks = predictions.masks.xy
        for pred in masks:
            # Remove padding from predictions and frame
            pred[:, 1] = pred[:, 1] - vertical_padding

            # Rescale to account for original camera size
            pred[:] = pred * scale

        # todo check confidence
        classes = [int(x) for x in predictions.boxes.cls.numpy().tolist()]

        # Publish the bounding boxes
        await CLIENT.publish_async("vision/segmentation", {
            "frame_id": data.get("frame_id"),
            "masks": np.concatenate(masks, axis=0),
            "classes": json.dumps(classes),
            "mask-lengths": json.dumps([x.shape[0] for x in masks]),
        })


if __name__ == "__main__":
    asyncio.run(main())
