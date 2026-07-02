# Camera Sensor

The Camera sensor service captures video from the default camera device, processes the frames, and publishes them to the Fabric.

## Features
- Captures video using OpenCV.
- Resizes frames to 920x512.
- Encodes frames as JPEG (Base64) for transport over the Fabric.
- Publishes to the `sensors/camera` stream.

## Requirements
- OpenCV (`opencv-python`)
- A working camera device (index 0).

## Running the Camera Service

Ensure the **Fabric** service is running first. Then, run the camera service:

```bash
python3 belle_bot/sensors/cameras/service.py
```

## Data Format

Messages are published to `sensors/camera` with the following JSON structure:

```json
{
    "rgb": "<base64_encoded_jpeg_string>",
    "shape": "[height, width, channels]"
}
```
