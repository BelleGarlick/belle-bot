import asyncio
import base64
import json
import time

from insightface.app import FaceAnalysis

from belle_bot.sensors import cameras
from belle_bot.fabric import FabricClient
from belle_bot.sensors.cameras.utils import parse_camera_stream

frame_queue = None
CLIENT = FabricClient()
loop = None

app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# todo make this testable

def show_frame_callback(data):
    loop.call_soon_threadsafe(frame_queue.put_nowait, data)


async def main():
    global frame_queue, loop
    frame_queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    CLIENT.listen(cameras.config.FABRIC_ID, show_frame_callback)

    print("Object Detection Running")
    while True:

        # Check if there is a new frame in the queue
        # In async, we wait for at least one item
        data = await frame_queue.get()

        # Only process the latest frame and drop everything else in the queue
        while not frame_queue.empty():
            data = frame_queue.get_nowait()

        start_time = time.time()

        frame = parse_camera_stream(data["rgb"])

        # Run predictions
        faces = app.get(frame, det_metric=0.3)

        detections = []
        for i, face in enumerate(faces):
            detections.append({
                "embedding": base64.b64encode(face.embedding.tobytes()).decode("utf-8"),
                "confidence": float(face.det_score),
                "bbox": base64.b64encode(face.bbox.tobytes()).decode("utf-8"),
            })

        end_time = time.time()

        # todo talk to the personal knowledge to work out which face is which

        await CLIENT.publish_async("vision/facial-recognition", {
            "frame_id": data.get("frame_id"),
            "faces": json.dumps(detections),
            "__duration": (end_time - start_time) * 1000,  # show performance in milliseconds
        })


if __name__ == "__main__":
    asyncio.run(main())
