import asyncio
import json

from ultralytics import YOLO

from belle_bot.sensors import cameras
from belle_bot.fabric import FabricClient
from belle_bot.sensors.cameras.utils import parse_camera_stream
from belle_bot.vision.utils import letterbox_image

frame_queue = None
CLIENT = FabricClient()
loop = None


model = YOLO("yolo26n-pose.pt")

# todo make this testable

def show_frame_callback(data):
    loop.call_soon_threadsafe(frame_queue.put_nowait, data)


def predict_bounding_boxes(frame, confidence_threshold=0.2):
    # r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    # frame = np.concatenate([np.expand_dims(x, axis=0) for x in (r, g, b)], axis=0)

    # frame = np.expand_dims(frame / 255, axis=0).astype(np.float32)

    # predictions = ort_sess.run(None, {"images": frame})[0][0]

    # todo switch this all out eventually when using onnx
    results = model(frame)

    return results[0].keypoints.data.numpy()


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

        # Apply padding so it's square for the image
        frame, scale, vertical_padding = letterbox_image(frame, size=640)

        # Run predictions
        predictions = predict_bounding_boxes(frame)

        # Remove padding from predictions and frame
        predictions[:, :, 1] = predictions[:, :, 1] - vertical_padding

        # Rescale to account for original camera size
        predictions[:, :, :] = predictions[:, :, :] * scale

        # Publish the bounding boxes
        await CLIENT.publish_async("vision/pose-estimation", {
            "frame_id": data.get("frame_id"),
            "predictions": predictions,
            "shape": json.dumps(predictions.shape),
        })


if __name__ == "__main__":
    asyncio.run(main())