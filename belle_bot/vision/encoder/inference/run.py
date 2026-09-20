import base64
import io
import json
import os
import random

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

from belle_bot.utils.cli import clpy
from belle_bot.vision.encoder.config.vision_encoder_video_config import VisionEncoderTrainingConfig
from belle_bot.vision.encoder.training.ml_model import VAE

path = "/Users/belle/Developer/belle-bot/belle_bot/mapping/positioning/training/environment/replays_cache/"


def _parse_events():
    # Filter out hidden files or non-replay files
    replay_ids = [f for f in os.listdir(path) if not f.startswith('.')]
    if not replay_ids:
        raise FileNotFoundError(f"No valid replay files found in {path}")

    random.shuffle(replay_ids)

    events = []
    for replay_id in replay_ids:
        replay_file = os.path.join(path, replay_id)

        try:
            with open(replay_file) as f:
                lines = f.readlines()
        except IsADirectoryError:
            continue

        for line in lines:
            if "," not in line:
                continue

            split_tokens = line.split(",")
            stream = split_tokens[0]

            if stream == "sensors/camera":
                try:
                    data = json.loads(",".join(split_tokens[2:]))
                    image_bytes = base64.b64decode(data['rgb'])
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    events.append(image)
                except (json.JSONDecodeError, KeyError):
                    continue

        # Exit early once a replay containing camera events is found
        if events:
            break

    return events


DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

config = clpy.parse_cli_args(VisionEncoderTrainingConfig())

models = [
    VAE(latent_dim=config.model.embedding_size),
]

for model in models:
    model.to(DEVICE)
    model.eval()

models[0].load_state_dict(torch.load("model-410000.pt", map_location=DEVICE))


def preprocess_frames(pil_imgs: list[Image.Image], device: torch.device) -> torch.Tensor:
    """Preprocesses a list of PIL Images into a normalized NCHW PyTorch tensor batch."""
    images = []
    for pil_img in pil_imgs:
        resized = pil_img.resize((224, 224))
        image = np.asarray(resized, dtype=np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        images.append(image)

    return torch.tensor(np.stack(images), dtype=torch.float32).to(device)


def predict_frames(model, tensor):
    with torch.no_grad():
        encoded, _ = model.encode(tensor)
        images = model.decode(encoded)

    encoded = encoded.cpu().detach().numpy()
    images = images.cpu().detach().numpy()

    # NCHW -> NHWC
    images = np.transpose(images, (0, 2, 3, 1))

    return encoded, images


if __name__ == "__main__":
    events = _parse_events()
    output_path = 'belle-bot-vision-encoder.mp4'

    all_frames = []
    batch_size = 32

    for i in range(0, len(events), batch_size):
        batch_events = events[i:i + batch_size]
        print(f"\rProcessing frames {i + 1}-{min(i + batch_size, len(events))}/{len(events)}", end="", flush=True)

        # 1. Prepare original frames for side-by-side display
        batch_orig_resized = [np.asarray(event.resize((224, 224)), dtype=np.float32) / 255.0 for event in batch_events]
        
        # Initialize list of lists for stacking: [frame_idx][model_idx]
        batch_frame_images = [[orig] for orig in batch_orig_resized]

        # 2. Preprocess batch for VAE model
        tensor = preprocess_frames(batch_events, DEVICE)

        # 3. Predict & overlay latent space representation
        for model in models:
            batch_encoded, batch_decoded_imgs = predict_frames(model, tensor)

            for j in range(len(batch_events)):
                encoded = batch_encoded[j]
                decoded_img = batch_decoded_imgs[j]

                # Reshape latent vector into a 32x32 single-channel block (32 * 32 = 1024)
                encoded_vis = np.reshape(encoded, (32, 32))
                encoded_vis = np.expand_dims(encoded_vis, axis=-1)
                encoded_vis = np.repeat(encoded_vis, 3, axis=-1)

                # Overlay latent grid onto bottom-center of decoded image
                h, w, _ = decoded_img.shape
                hh, hw = 5 * (h // 6), w // 2
                decoded_img[hh - 16:hh + 16, hw - 16:hw + 16] = encoded_vis

                batch_frame_images[j].append(decoded_img)

        # 4. Concatenate original and reconstructed frames horizontally for the batch
        for frame_images in batch_frame_images:
            render_image = np.hstack(frame_images)
            render_image = np.clip(render_image * 255, 0, 255).astype(np.uint8)
            all_frames.append(render_image)

    # 5. Write stacked frames to disk via imageio
    if all_frames:
        iio.imwrite(output_path, np.stack(all_frames), fps=20)
        print(f"\nVideo saved successfully to {output_path} ({len(all_frames)} total frames processed)")
    else:
        print("\nNo frames were processed.")