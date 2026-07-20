import json
import time
import uuid

import cv2
import numpy as np

from belle_bot.infra.fabric import FabricClient

CLIENT = FabricClient()
FABRIC_ID = "sensors/camera"
MAX_FPS = 10
JPEG_QUALITY = 60

# how to encode the depth data. it's not used in the system, just for roboviz. doesn't need to be as high
DEPTH_JPEG_QUALITY = 40

if __name__ == "__main__":
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    profile = pipe.start()

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Set queue size to 2 on all sensors to minimize latency/lag
    device = profile.get_device()
    for sensor in device.query_sensors():
        if sensor.supports(rs.option.frames_queue_size):
            sensor.set_option(rs.option.frames_queue_size, 2)

    # Initialize hole filling filter (Nearest neighbor mode)
    hole_filler = rs.hole_filling_filter(1)

    # Initialize the colorizer tool to convert 16-bit depth to 8-bit RGB
    colorizer = rs.colorizer()
    # Optional: Choose a specific color scheme (e.g., 0 = Jet, 2 = Grayscale 8-bit)
    # colorizer.set_option(rs.option.visual_preset, 2)

    try:
        target_interval = 1.0 / MAX_FPS
        while True:
            start_time = time.perf_counter()
            frames = pipe.wait_for_frames()

            # Flush the queue to keep only the absolute latest frameset
            while True:
                extra_frames = pipe.poll_for_frames()
                if not extra_frames:
                    break
                frames = extra_frames

            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Check if frames are valid
            if not depth_frame or not color_frame:
                continue

            # Fill missing depth values
            depth_frame = hole_filler.process(depth_frame)

            # Convert depth frame to an 8-bit RGB colorized frame
            colorized_depth_frame = colorizer.colorize(depth_frame)

            # Convert both to numpy arrays
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_raw = cv2.resize(depth_raw, tuple([x//2 for x in depth_raw.shape]))
            depth_image_rgb = np.asanyarray(colorized_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Convert both from RGB (RealSense) to BGR (OpenCV)
            depth_bgr = cv2.cvtColor(depth_image_rgb, cv2.COLOR_RGB2BGR)
            color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Compress both as standard 8-bit BGR images
            color_encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            _, color_buffer = cv2.imencode('.jpg', color_bgr, color_encode_param)

            depth_encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), DEPTH_JPEG_QUALITY]
            _, depth_buffer = cv2.imencode('.jpg', depth_bgr, depth_encode_param)

            CLIENT.publish(FABRIC_ID, {
                "service_name": FABRIC_ID,
                "frame_id": str(uuid.uuid4()),
                "rgb": color_buffer,
                "depth_raw": depth_raw,
                "depth": depth_buffer,
                "shape": json.dumps(color_image.shape),
                "jpeg_quality": JPEG_QUALITY,
            })

            # Rate limit the publishing rate to match config.MAX_FPS
            elapsed_time = time.perf_counter() - start_time
            sleep_time = max(0.0, target_interval - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        pipe.stop()
