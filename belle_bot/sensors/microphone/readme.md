# Microphone Sensor

The Microphone sensor service captures audio from the default input device and publishes it to the Fabric.

## Features
- Captures audio using `sounddevice`.
- Buffers audio into 0.5s chunks (by default).
- Encodes audio as Base64 (int16) for transport over the Fabric.
- Publishes to the `sensor/microphone` stream.

## Requirements
- `sounddevice`
- `numpy`
- PortAudio (system dependency for `sounddevice`)

## Configuration
Configuration is managed in `belle_bot/sensors/microphone/config.py`:
- `SAMPLE_RATE`: Default is 16000 Hz.
- `BUFFER_SIZE`: Default is 0.5s of audio samples.

## Running the Microphone Service

Ensure the **Fabric** service is running first. Then, run the microphone service:

```bash
python3 belle_bot/sensors/microphone/service.py
```

## Data Format

Messages are published to `sensor/microphone` with the following JSON structure:

```json
{
    "audio": "<base64_encoded_int16_audio_data>",
    "sample_rate": "16000",
    "channels": "1"
}
```
