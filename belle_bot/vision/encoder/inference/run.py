import base64
import json
import random
from typing import Literal

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch

from belle_bot.utils.cli import clpy
from belle_bot.vision.encoder.config.vision_encoder_video_config import VisionEncoderTrainingConfig
from belle_bot.vision.encoder.training.data_loader import custom_decoder, merge
from houston.client.py import replays


def get_replay_ids(subset: Literal["train", "eval"] | None, count: int = 6):
    filter_tags = ["dataset/vision/encoder"]
    if subset:
        filter_tags.append(subset)

    replay_ids = replays.query_replays(
        config.houston,
        page=0,
        tags=filter_tags
    )['replays']

    all_ids = sorted([x["replay_id"] for x in replay_ids])
    if len(all_ids) < count:
        print(f"Warning: Requested {count} replays, but only found {len(all_ids)}.")
        return all_ids

    return random.sample(all_ids, count)


def _parse_replays(num_replays: int = 6):
    replay_ids = get_replay_ids("train", count=num_replays)
    
    all_replays_events = []
    for replay_id in replay_ids:
        replay_file = replays.get_replay_file(config.houston, replay_id)
        print(f"Loading replay: {replay_id}")

        events = []
        lines = replay_file.split("\n")
        for line in lines:
            if "," not in line:
                continue

            split_tokens = line.split(",")
            stream = split_tokens[0]

            if stream == "sensors/camera":
                try:
                    data = json.loads(",".join(split_tokens[2:]))
                    events.append({
                        'rgb': base64.b64decode(data['rgb']),
                        'depth': base64.b64decode(data['depth']),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

        if events:
            all_replays_events.append(events)

    if not all_replays_events:
        return []

    # Trim all replays to the length of the shortest sequence so they play in sync
    min_length = min(len(events) for events in all_replays_events)
    print(f"Synchronizing {len(all_replays_events)} replays to shortest sequence length: {min_length} frames")

    return [events[:min_length] for events in all_replays_events]


DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

config = clpy.parse_cli_args(VisionEncoderTrainingConfig())

# Load ONNX models
encoder_path = "model_encoder_small.onnx"
decoder_path = "model_decoder_small.onnx"

providers = ['CPUExecutionProvider']
if torch.backends.mps.is_available():
    providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
elif torch.cuda.is_available():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

ort_encoder = ort.InferenceSession(encoder_path, providers=providers)
ort_decoder = ort.InferenceSession(decoder_path, providers=providers)


def tensor_frame_to_joint_frame(t):
    rgb_t = t[:, :, :3]
    depth_t = t[:, :, 3]
    depth_max = np.max(depth_t)
    if depth_max > 0:
        depth_t = depth_t / depth_max

    cm = plt.get_cmap('rainbow')
    depth_colored = cm(depth_t)[:, :, :3]  # Remove alpha channel

    return np.concatenate((rgb_t, depth_colored), axis=1)


def process_event_batch(batch_events):
    """Encodes and decodes a list/batch of events, returning visual composite frames."""
    image_pairs = [custom_decoder(x) for x in batch_events]
    merged_frames = np.array([merge(x) for x in image_pairs])
    joint_input_frames = [tensor_frame_to_joint_frame(x) for x in merged_frames]
    
    merged_tensor = np.transpose(merged_frames, (0, 3, 1, 2)).astype(np.float32)

    ort_inputs = {ort_encoder.get_inputs()[0].name: merged_tensor}
    batch_encoded = ort_encoder.run(None, ort_inputs)[0]

    ort_inputs_dec = {ort_decoder.get_inputs()[0].name: batch_encoded}
    batch_images = ort_decoder.run(None, ort_inputs_dec)[0]
    batch_images = np.transpose(batch_images, (0, 2, 3, 1))

    processed_frames = []
    for j in range(len(batch_events)):
        encoded = batch_encoded[j]
        decoded_img = batch_images[j]

        encoded_vis = np.vstack((
            np.clip(encoded, 0, 100),
            np.clip(encoded, 0, 100),
            np.clip(-encoded, 0, 100)
        )).reshape((27, 27, 3))
        encoded_vis = cv2.resize(encoded_vis, (50, 50), interpolation=cv2.INTER_NEAREST)
        encoded_vis = encoded_vis * 100

        joint_frame_in = np.array(joint_input_frames[j] * 255, dtype=np.uint8)
        joint_frame = np.array(tensor_frame_to_joint_frame(decoded_img) * 255, dtype=np.uint8)
        ref = np.concatenate((joint_frame_in, joint_frame), axis=0)

        # Overlay latent grid onto center of image
        h, w, _ = ref.shape
        hh, hw = h // 2, w // 2
        ref[hh - 25:hh + 25, hw - 25:hw + 25] = encoded_vis

        processed_frames.append(ref)

    return processed_frames


if __name__ == "__main__":
    NUM_REPLAYS = 3
    replays_events = _parse_replays(num_replays=NUM_REPLAYS)
    output_path = 'belle-bot-vision-encoder-grid.mp4'

    if not replays_events:
        print("No valid replays found.")
        exit()

    num_frames = len(replays_events[0])
    num_streams = len(replays_events)
    batch_size = 50

    # Process all replays frame sequence by batch
    stream_frames = [[] for _ in range(num_streams)]

    for i in range(0, num_frames, batch_size):
        end_idx = min(i + batch_size, num_frames)
        print(f"\rProcessing frames {i + 1}-{end_idx}/{num_frames} across {num_streams} replays", end="", flush=True)

        for s_idx in range(num_streams):
            batch_events = replays_events[s_idx][i:end_idx]
            out_frames = process_event_batch(batch_events)
            stream_frames[s_idx].extend(out_frames)

    print("\nCompositing frames into 3x2 grid...")
    grid_frames = []
    
    for t in range(num_frames):
        current_t_frames = [stream_frames[s_idx][t] for s_idx in range(num_streams)]
        
        # Pad with black frames if fewer than 6 replays are returned
        while len(current_t_frames) < 6:
            current_t_frames.append(np.zeros_like(current_t_frames[0]))

        # Stack into 3x2 grid
        grid_frame = np.hstack(current_t_frames[0:3])
        #row2 = np.hstack(current_t_frames[3:6])
        #grid_frame = np.vstack((row1, row2))

        grid_frames.append(grid_frame)

    if grid_frames:
        iio.imwrite(output_path, np.stack(grid_frames), fps=20)
        print(f"Video saved successfully to {output_path} ({len(grid_frames)} grid frames)")
    else:
        print("No frames were processed.")
