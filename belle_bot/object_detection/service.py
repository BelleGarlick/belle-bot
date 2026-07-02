import asyncio
import cv2
import numpy as np

import onnxruntime as ort

from belle_bot.sensors import cameras
from belle_bot.fabric import FabricClient
from belle_bot.sensors.cameras.utils import parse_camera_stream

frame_queue = None
CLIENT = FabricClient()
loop = None

ort_sess = ort.InferenceSession("best.onnx")

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
    r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    frame = np.concatenate([np.expand_dims(x, axis=0) for x in (r, g, b)], axis=0)

    frame = np.expand_dims(frame / 255, axis=0).astype(np.float32)

    predictions = ort_sess.run(None, {"images": frame})[0][0]

    return predictions[np.where(predictions[:, 4] > confidence_threshold)]


async def main():
    global frame_queue, loop
    frame_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    CLIENT.listen(cameras.config.FABRIC_ID, show_frame_callback)

    while True:
        print("Object Detection Running")

        # Check if there is a new frame in the queue
        # In async, we wait for at least one item
        data = await frame_queue.get()

        # Only process the latest frame and drop everything else in the queue
        while not frame_queue.empty():
            data = frame_queue.get_nowait()

        frame = parse_camera_stream(data["rgb"])

        # Resize the image to the height needed for the yolo model which we can then split up
        frame, scale = rescale_frame(frame)

        # Apply padding so it's square for the image
        frame, vertical_padding = reframe_image(frame)

        # Run predictions
        predictions = predict_bounding_boxes(frame)

        # Remove padding from predictions and frame
        predictions[:, 1] = predictions[:, 1] - vertical_padding
        predictions[:, 3] = predictions[:, 3] - vertical_padding

        # Rescale to account for original camera size
        predictions[:, :4] = predictions[:, :4] * scale

        # Publish the bounding boxes
        await CLIENT.publish_async("vision/bounding_boxes", {
            "frame_id": data.get("frame_id"),
            "predictions": predictions
        })


if __name__ == "__main__":
    asyncio.run(main())