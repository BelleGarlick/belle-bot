import json
import time
import uuid

import cv2
import numpy as np

from belle_bot.fabric import FabricClient

CLIENT = FabricClient()
FABRIC_ID = "sensors/camera"
MAX_FPS = 10
JPEG_QUALITY = 60

# how to encode the depth data. it's not used in the system, just for roboviz. doesn't need to be as high
DEPTH_JPEG_QUALITY = 40

color_encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
depth_encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), DEPTH_JPEG_QUALITY]

if __name__ == "__main__":
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    profile = pipe.start()

    align_to = rs.stream.color
    align = rs.align(align_to)
    # 2. Hardware Settings & Presets
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()

    # Load high accuracy visual preset (0 = Custom, 1 = Default, 2 = Hand, 3 = High Accuracy, 4 = High Density)
    if depth_sensor.supports(rs.option.visual_preset):
        depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy

    # Enable Laser Emitter for better stereo matching on plain surfaces
    if depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1)

    for sensor in device.query_sensors():
        if sensor.supports(rs.option.frames_queue_size):
            sensor.set_option(rs.option.frames_queue_size, 2)

    # 3. Instantiate Post-Processing Filters in Recommended Order
    decimation = rs.decimation_filter(magnitude=1)  # Set to 2 if you want 2x hardware downscaling
    # threshold = rs.threshold_filter(min_dist=0.15, max_dist=4.0)  # Adjust min/max meters for your scene
    depth_to_disparity = rs.disparity_transform(True)

    # spatial filtering
    spatial = rs.spatial_filter(
        smooth_alpha=0.5,
        smooth_delta=20,
        magnitude=2,
        hole_fill=2
    )

    temporal = rs.temporal_filter(smooth_alpha=0.4, smooth_delta=20, persistence_control=3)
    disparity_to_depth = rs.disparity_transform(False)

    # Standalone hole filler using Mode 2 (nearest_from_around) instead of Mode 1
    hole_filler = rs.hole_filling_filter(mode=2)

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

            # filtered_depth = decimation.process(depth_frame)
            # filtered_depth = threshold.process(filtered_depth)
            # filtered_depth = depth_to_disparity.process(filtered_depth)
            # filtered_depth = spatial.process(filtered_depth)
            # filtered_depth = temporal.process(filtered_depth)
            # filtered_depth = disparity_to_depth.process(filtered_depth)
            # filtered_depth = hole_filler.process(filtered_depth)

            # Convert depth frame to an 8-bit RGB colorized frame
            # colorized_depth_frame = colorizer.colorize(filtered_depth)

            # Convert both from RGB (RealSense) to BGR (OpenCV)
            color_image = np.asanyarray(color_frame.get_data())  # colour image
            # depth_raw = np.asanyarray(depth_frame.get_data())  # raw data
            depth = np.asanyarray(depth_frame.get_data())  # processed

            # convert colour for colour one
            color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # Resize the depth image since the full definition is not as important
            h, w = depth.shape[:2]
            depth = cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)

            # Create a depth-preview
            depth_preview = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_preview = np.uint8(depth_preview)
            # depth_raw_preview = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX)
            # depth_raw_preview = np.uint8(depth_raw_preview)

            # Encode the images
            # todo document how to read this data elsewhere
            _, depth_frame = cv2.imencode('.png', depth)
            _, depth_preview = cv2.imencode('.jpg', depth_preview, depth_encode_param)
            _, color_buffer = cv2.imencode('.jpg', color_bgr, color_encode_param)

            CLIENT.publish(FABRIC_ID, {
                "service_name": FABRIC_ID,
                "frame_id": str(uuid.uuid4()),
                "rgb": color_buffer,
                "depth": depth_frame,
                "depth_preview": depth_preview,
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
