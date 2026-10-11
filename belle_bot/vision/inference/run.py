import base64
import json
import os
import random
from typing import Literal

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch

from belle_bot.utils.cli import clpy
from belle_bot.vision import tasks
from belle_bot.vision.config.vision_encoder_video_config import VisionEncoderVideoConfig
from belle_bot.vision.tasks.bounding_boxes.model import VisualCortexBoundingBoxModel
from belle_bot.vision.tasks.encoder.data_loader import extract_encoder_input, merge
from belle_bot.vision.tasks.encoder.model import VisualCortexEncoder
from belle_bot.vision.tasks.segmentation.model import VisualCortexSegmentationDecoder
from houston.client.py import replays


def get_replay_ids(subset: Literal["train", "eval"] | None, count: int = 6):
    all_ids = [x for x in os.listdir("/Users/belle/Developer/belle-bot/belle_bot/mapping/positioning/training/environment/replays_cache") if x[0] != "."]

    all_ids.remove("049e17b1-9646-4fc6-a41c-e7bf651b29ae.txt")
    all_ids.remove("2adcc988-f08f-49ac-9262-1aeaa68e30f6.txt")
    all_ids.remove("3f4c9b31-4a76-4dbf-a428-b42ca75bedfd.txt")
    all_ids.remove("437812d0-a336-4481-95a8-7d169a0c4d97.txt")
    all_ids.remove("4bc42bcc-1d36-4a57-b5eb-9703f96f66fe.txt")
    all_ids.remove("768641ed-3420-4a73-a470-0a3aa87a1662.txt")
    all_ids.remove("90f73f20-0ab0-4cad-b8b9-778639a33f8e.txt")
    all_ids.remove("b233341f-425e-4cf1-8091-ee528862d9dc.txt")
    all_ids.remove("b6115386-006f-4213-ad6e-dae65655bf9c.txt")

    return random.sample(all_ids, count)

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
        with open("/Users/belle/Developer/belle-bot/belle_bot/mapping/positioning/training/environment/replays_cache/" + replay_id) as file:
            replay_file = file.read()

        # replay_file = replays.get_replay_file(config.houston, replay_id)
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

config = clpy.parse_cli_args(VisionEncoderVideoConfig())
config.model.encoder.file_path = "visual_cortex_v2b1_encoder.pt"
config.model.decoder.file_path = "visual_cortex_v2b1_decoder.pt"
config.model.segmentation.file_path = "visual_cortex_v2b1_segmentation.pt"
config.model.detection.file_path = "visual_cortex_v2b1_bounding_box.pt"

tasks = {
    "encoder": VisualCortexEncoder().to(DEVICE),
    "decoder": tasks.decoder.model.VisualCortexDecoder().to(DEVICE),
    "segmentation": VisualCortexSegmentationDecoder().to(DEVICE),
    "detection": VisualCortexBoundingBoxModel().to(DEVICE),
}

tasks['encoder'].load_state_dict(torch.load(config.model.encoder.file_path, map_location=DEVICE))
tasks['decoder'].load_state_dict(torch.load(config.model.decoder.file_path, map_location=DEVICE))
tasks['segmentation'].load_state_dict(torch.load(config.model.segmentation.file_path, map_location=DEVICE))
tasks['detection'].load_state_dict(torch.load(config.model.detection.file_path, map_location=DEVICE))


def render_depth(depth_t):
    depth_max = np.max(depth_t)
    if depth_max > 0:
        depth_t = depth_t / depth_max

    cm = plt.get_cmap('rainbow')
    return np.array(cm(depth_t)[:, :, :3] * 255, dtype=np.uint8)


def process_event_batch(batch_events):
    """Encodes and decodes a list/batch of events, returning visual composite frames."""
    image_pairs = [extract_encoder_input(x) for x in batch_events]
    merged_frames = np.array([merge(x) for x in image_pairs])

    merged_tensor = torch.tensor(merged_frames.transpose((0, 3, 1, 2)), device=DEVICE)

    with torch.no_grad():
        for model in tasks.values():
            model.eval()

        encoding = tasks['encoder'](merged_tensor)[0]
        bounding_boxes = tasks['detection'](encoding)
        segmentation = tasks['segmentation'](encoding)
        decoding = tasks['decoder'](encoding)

    return {
        "input": merged_frames,
        "encoding": encoding.detach().cpu().numpy(),
        "bounding_boxes": bounding_boxes.detach().cpu().numpy(),
        "segmentation": segmentation.detach().cpu().numpy(),
        "decoding": decoding.detach().cpu().numpy().transpose((0, 2, 3, 1))
    }


def render_outputs(batch):
    rendered_outputs = []

    for i in range(len(batch['encoding'])):
        encoded = batch['encoding'][i]
        decoded_img = batch['decoding'][i]
        decoded_seg = batch['segmentation'][i]

        embedding = np.vstack((
            np.clip(encoded, 0, 100),
            np.clip(encoded, 0, 100),
            np.clip(-encoded, 0, 100)
        )).reshape((32, 32, 3)) * 100

        seg_frame = np.argmax(decoded_seg, axis=0)
        seg_frame = render_depth(seg_frame / 80)

        rendered_outputs.append({
            "embedding": embedding,
            "input_rgb": np.array(batch['input'][i][:, :, :3] * 255, dtype=np.uint8),
            "output_rgb": np.array(decoded_img[:, :, :3] * 255, dtype=np.uint8),
            "output_depth": render_depth(decoded_img[:, :, -1]),
            "output_segmentation": seg_frame
        })

    return rendered_outputs


def render_event_batch(batch):
    processed_frames = []
    for item in batch:
        encoded_vis = cv2.resize(item['embedding'], (50, 50), interpolation=cv2.INTER_NEAREST)

        # todo render bounding boxes
        ref = np.vstack((
            np.hstack((item['input_rgb'], item['output_rgb'])),
            np.hstack((item['output_depth'], item['output_segmentation']))
        ))

        # Overlay latent grid onto center of image
        h, w, _ = ref.shape
        hh, hw = h // 2, w // 2
        ref[hh - 25:hh + 25, hw - 25:hw + 25] = encoded_vis

        ref = cv2.resize(ref, (400, 400), interpolation=cv2.INTER_NEAREST)

        processed_frames.append(ref)

    return processed_frames


if __name__ == "__main__":
    NUM_REPLAYS = 3
    replays_events = _parse_replays(num_replays=NUM_REPLAYS)
    output_path = 'belle-bot-vision-encoder-grid.mp4'

    if not replays_events:
        print("No valid replays found.")
        exit()

    num_frames = len(replays_events[0]) // 5
    num_streams = len(replays_events)
    batch_size = 50

    # Process all replays frame sequence by batch
    stream_frames = [[] for _ in range(num_streams)]

    # Feed through ml models
    for i in range(0, num_frames, batch_size):
        end_idx = min(i + batch_size, num_frames)

        for s_idx in range(num_streams):
            print(f"\rProcessing frames {s_idx} {i + 1}-{end_idx}/{num_frames} across {num_streams} replays", end="", flush=True)
            batch_events = replays_events[s_idx][i:end_idx]
            predictions = process_event_batch(batch_events)
            rendered_outputs = render_outputs(predictions)
            out_frames = render_event_batch(rendered_outputs)
            stream_frames[s_idx].extend(out_frames)

    print("\nCompositing frames into 3x2 grid...")
    grid_frames = []
    
    for t in range(num_frames):
        current_t_frames = [stream_frames[s_idx][t] for s_idx in range(num_streams)]
        
        # Pad with black frames if fewer than 6 replays are returned
        while len(current_t_frames) < NUM_REPLAYS:
            current_t_frames.append(np.zeros_like(current_t_frames[0]))

        # Stack into 3x2 grid
        grid_frame = np.hstack(current_t_frames[0:3])
        # row2 = np.hstack(current_t_frames[3:6])
        # grid_frame = np.vstack((row1, row2))

        grid_frames.append(grid_frame)

    if grid_frames:
        iio.imwrite(output_path, np.stack(grid_frames), fps=20)
        print(f"Video saved successfully to {output_path} ({len(grid_frames)} grid frames)")
    else:
        print("No frames were processed.")
