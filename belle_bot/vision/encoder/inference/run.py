import base64
import json
import os
import random

import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch

from belle_bot.utils.cli import clpy
from belle_bot.vision.encoder.config.vision_encoder_video_config import VisionEncoderTrainingConfig
from belle_bot.vision.encoder.training.data_loader import custom_decoder, merge
from belle_bot.vision.encoder.training.ml_model import VAE2_448
from houston.client.py import replays

path = "/Users/belle/Developer/belle-bot/downloaded_replays"


def get_replay_ids(subset: Literal["train", "eval"] | None):
    filter = ["dataset/vision/encoder"]
    if subset:
        filter += [subset]

    replay_ids = replays.query_replays(
        config.houston,
        page=0,
        tags=filter
    )['replays']

    return sorted([x["replay_id"] for x in replay_ids])


def _parse_events():
    # Filter out hidden files or non-replay files
    replay_ids = get_replay_ids("test")
    random.shuffle(replay_ids)

    events = []
    for replay_id in replay_ids:
        replay_file = replays.get_replay_file(config.houston, replay_id)
        print(replay_id)

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
            break

    # events = events[:100] + events[-100:]

    return events


DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

config = clpy.parse_cli_args(VisionEncoderTrainingConfig())

models = [VAE2_448(img_channels=4, latent_dim=config.model.embedding_size)]
# models = [VAE2_448(img_channels=4, latent_dim=484)]

for model in models:
    model.to(DEVICE)
    model.eval()

models[0].load_state_dict(torch.load("model-729_2.pt", map_location=DEVICE))


def predict_frames(model, tensor):
    with torch.no_grad():
        encoded, _ = model.encode(tensor)
        images = model.decode(encoded)

    encoded = encoded.cpu().detach().numpy()
    images = images.cpu().detach().numpy()

    # NCHW -> NHWC
    images = np.transpose(images, (0, 2, 3, 1))

    return encoded, images


def tensor_frame_to_joint_frame(t):
    rgb_t = t[:, :, :3]
    depth_t = t[:, :, 3]
    depth_t = depth_t / np.max(depth_t)

    # Use a colormap to colorize depth (e.g., 'viridis')
    # depth_t is expected to be in [0, 1] range
    cm = plt.get_cmap('rainbow')
    depth_colored = cm(depth_t)[:, :, :3]  # Remove alpha channel if present

    return np.concatenate((rgb_t, depth_colored), axis=1)


if __name__ == "__main__":
    events = _parse_events()
    output_path = 'belle-bot-vision-encoder.mp4'

    all_frames = []
    batch_size = 1

    encodeds = []

    for i in range(0, len(events), batch_size):
        batch_events = events[i:i + batch_size]
        print(f"\rProcessing frames {i + 1}-{min(i + batch_size, len(events))}/{len(events)}", end="", flush=True)

        image_pairs = [custom_decoder(x) for x in batch_events]
        merged_frames = np.array([merge(x) for x in image_pairs])
        joint_input_frames = [tensor_frame_to_joint_frame(x) for x in merged_frames]
        merged_frames = np.transpose(merged_frames, (0, 3, 1, 2))
        tensors = torch.tensor(merged_frames).to(DEVICE)

        for model in models:
            with torch.no_grad():
                encoded, _ = model.encode(tensors)
                images = model.decode(encoded)

            batch_encoded = encoded.cpu().detach().numpy()
            batch_images = images.cpu().detach().numpy()
            batch_images = np.transpose(batch_images, (0, 2, 3, 1))

            for j in range(len(batch_events)):
                encoded = batch_encoded[j]
                decoded_img = batch_images[j]

                # Reshape latent vector into a 32x32 single-channel block (32 * 32 = 1024)
                # encoded_vis = np.reshape(encoded, (27, 27))
                # encoded_vis = np.expand_dims(encoded_vis, axis=-1)
                # encoded_vis = np.repeat(encoded_vis, 3, axis=-1)

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


                # todo change the bright colours around
                # encoded_vis = (encoded_vis * 10) + 128
                # encoded_vis = encoded_vis / np.max(encoded_vis)

                encodeds.append(encoded)

                # Overlay latent grid onto bottom-center of decoded image
                h, w, _ = ref.shape
                hh, hw = h // 2, w // 2
                ref[hh - 25:hh + 25, hw - 25:hw + 25] = encoded_vis

                all_frames.append(ref)

        # # 4. Concatenate original and reconstructed frames horizontally for the batch
        # for frame_images in batch_frame_images:
        #     render_image = np.hstack(frame_images)
        #     render_image = np.clip(render_image * 255, 0, 255).astype(np.uint8)
        #     all_frames.append(render_image)

    # 5. Write stacked frames to disk via imageio
    if all_frames:
        iio.imwrite(output_path, np.stack(all_frames), fps=20)
        print(f"\nVideo saved successfully to {output_path} ({len(all_frames)} total frames processed)")
    else:
        print("\nNo frames were processed.")

    print(np.array(encodeds).max())
    print(np.array(encodeds).min())
