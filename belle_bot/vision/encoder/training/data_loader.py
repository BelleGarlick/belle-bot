import io
from pathlib import Path

import cv2
import numpy as np
import torch
import webdataset as wds
from PIL import Image


# Dataset setup
base_path = Path("/run/media/belle/Houston/datasets/vision-encoder/v1/")
train_path = str("/Users/belle/Developer/belle-bot/train-{000000..000000}.tar")
# train_path = str(base_path / "train-{000000..000004}.tar")
test_path = str(base_path / "test-{000000..000001}.tar")


def custom_decoder(data):
    main_image_bytes = io.BytesIO(data['rgb'])
    depth_image_bytes = np.frombuffer(data['depth'], dtype=np.uint8)
    depth_mask = np.frombuffer(data['depth_mask'], dtype=np.uint8)

    # conver the images
    main_image = np.array(Image.open(main_image_bytes))
    depth_data = cv2.imdecode(depth_image_bytes, cv2.IMREAD_UNCHANGED)

    # rgb is (H, W, 3), depth is (H, W)
    main_image = cv2.resize(main_image, (448, 448))
    depth_data = cv2.resize(depth_data, (448, 448))
    depth_mask = depth_mask.reshape((448, 448))

    return main_image, depth_data, depth_mask


def jitter(data):
    # both are np arrays of 448x448
    rgb_t, depth_t, depth_mask = data
    im_size = rgb_t.shape[0]

    # randomly flip image
    if np.random.random() > 0.5:
        rgb_t = np.fliplr(rgb_t)
        depth_t = np.fliplr(depth_t)

    # crop image randomly
    random_crop_size = np.random.randint(int(im_size * 0.9), im_size)
    random_x_offset = np.random.randint(0, im_size - random_crop_size)
    random_y_offset = np.random.randint(0, im_size - random_crop_size)

    # crop the image at the offsets and resize
    rgb_t = rgb_t[random_x_offset:random_x_offset + random_crop_size, random_y_offset:random_y_offset + random_crop_size]
    depth_t = depth_t[random_x_offset:random_x_offset + random_crop_size, random_y_offset:random_y_offset + random_crop_size]
    if depth_mask is not None:
        depth_mask = depth_mask[random_x_offset:random_x_offset + random_crop_size, random_y_offset:random_y_offset + random_crop_size]

    rgb_t = cv2.resize(rgb_t, (448, 448))
    depth_t = cv2.resize(depth_t, (448, 448))
    if depth_mask is not None:
        depth_mask = cv2.resize(depth_mask, (448, 448))

    # random colour shift
    if np.random.random() > 0.2:
        # Brightness and contrast
        brightness = np.random.uniform(0.8, 1.2)
        contrast = np.random.uniform(0.8, 1.2)
        rgb_t = cv2.convertScaleAbs(rgb_t, alpha=contrast, beta=127 * (1 - contrast) + (brightness - 1) * 255)

        # Hue and saturation
        hsv = cv2.cvtColor(rgb_t, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-9, 9)) % 180  # Hue shift
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.8, 1.2), 0, 255)  # Saturation scale
        rgb_t = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return rgb_t, depth_t, depth_mask


def merge(data):
    rgb, depth, depth_mask = data

    # Convert to float32 and scale to [0, 1]
    rgb = rgb.astype(np.float32) / 255.0
    depth = depth.astype(np.float32) / 65535.0  # Assuming 16-bit depth
    depth = np.expand_dims(depth, axis=2)

    depth_input = depth.copy()
    depth_output = depth.copy()

    if depth_mask is not None:
        depth_input[np.where(depth_mask == 1)] = 0

    x = np.concatenate((rgb, depth_input), axis=-1)
    y = np.concatenate((rgb, depth_output), axis=-1)

    return x, y


def to_tensor(samples):
    x_batch = []
    y_batch = []
    for x, y in samples:
        # x, y are (H, W, C) -> (C, H, W)
        x_t = torch.from_numpy(x).permute(2, 0, 1)
        y_t = torch.from_numpy(y).permute(2, 0, 1)
        x_batch.append(x_t)
        y_batch.append(y_t)
    return torch.stack(x_batch), torch.stack(y_batch)


def load_dataset(config):
    train_dataset = (
        wds.WebDataset(train_path, resampled=True)
        .shuffle(10000)
        .map(custom_decoder)
        .map(jitter)
        .map(merge)
        .batched(config.mini_batch_size, collation_fn=to_tensor)
    )

    test_dataset = (
        wds.WebDataset(test_path, resampled=True)
        .shuffle(10000)
        .map(custom_decoder)
        .map(merge)
        .batched(config.mini_batch_size, collation_fn=to_tensor)
    )

    return train_dataset, test_dataset
