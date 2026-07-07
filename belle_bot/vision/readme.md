# Vision Pipeline

The vision pipeline for belle-bot provides various real-time image processing capabilities including object detection, pose estimation, facial recognition, and image segmentation. All services communicate via the `Fabric` messaging system.

## Overview

The pipeline consists of several independent services that listen for camera frames and publish their respective detections.

### Services

| Service | Description | Model/Framework | Fabric Topic |
|---------|-------------|-----------------|--------------|
| **Object Detection** | Detects objects (specifically humans) and provides bounding boxes. | YOLO (ONNX) | `vision/bounding-boxes` |
| **Pose Estimation** | Identifies human keypoints for skeleton tracking. | YOLO Pose | `vision/pose-estimation` |
| **Facial Recognition**| Detects faces and generates embeddings for identification. | InsightFace | `vision/facial-recognition` |
| **Segmentation** | Provides pixel-level masks for detected objects. | YOLO Segmentation | `vision/segmentation` |

## Communication Flow

1.  **Ingestion**: Services listen to `sensors/camera` (defined in `belle_bot.sensors.cameras.config.FABRIC_ID`) for incoming RGB frames.
2.  **Preprocessing**: Frames are typically parsed using `parse_camera_stream` and resized/padded using `letterbox_image` to fit model requirements (usually 640x640).
3.  **Inference**: Models process the frame to produce detections.
4.  **Postprocessing**: Detections are rescaled back to the original camera dimensions.
5.  **Publication**: Results are published to their respective Fabric topics as JSON-encoded data.

## Directory Structure

- `object_detection/`: YOLO-based detection service and ONNX model.
- `pose_estimation/`: Human pose tracking service.
- `facial_recognition/`: Face detection and embedding generation.
- `segmentation_masks/`: Pixel-level image segmentation.
- `utils/`: Common utilities for image manipulation (e.g., letterboxing).

## Getting Started

### Running a Service

Each service can be started by running its `service.py` file. For example:

```bash
python -m belle_bot.vision.object_detection.service
```

### Visualizing Outputs

Some modules include a `watcher.py` script that provides a local CV2 window to preview the detections overlaid on the camera stream.

```bash
python -m belle_bot.vision.object_detection.watcher
```

## Dependencies

Major dependencies include:
- `opencv-python`: Image processing and visualization.
- `onnxruntime`: Efficient inference for object detection.
- `ultralytics`: YOLO implementation for pose and segmentation.
- `insightface`: Facial analysis and recognition.
- `numpy`: Numerical operations on image arrays.
