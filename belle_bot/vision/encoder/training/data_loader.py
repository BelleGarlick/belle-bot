import io
from pathlib import Path

import cv2
import numpy as np
import torch
import webdataset as wds
from PIL import Image


import torchvision.transforms as T
import torchvision.transforms.functional as F_vision
from matplotlib import pyplot as plt

# Dataset setup
base_path = Path("/run/media/belle/Houston/datasets/vision-encoder/v1/")
train_path = str(base_path / "train-{000000..000002}.tar")
test_path = str(base_path / "test-{000000..000001}.tar")


def preprocess(image, jitter=False, is_depth=False):
    import random

    # 1. Resize
    image = F_vision.resize(image, (448, 448))

    if jitter:
        # 2. Random Resized Crop
        # Note: We need to use the same parameters for both images in a pair.
        # This function should probably be called with params already calculated if we want perfect sync.
        # But for now, we'll try to use the same random seed in preprocess_train.
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(0.75, 1.33))
        image = F_vision.resized_crop(image, i, j, h, w, (448, 448))

        # 3. Random Horizontal Flip
        if random.random() > 0.5:
            image = F_vision.hflip(image)

        # 4. Color Jitter (only for RGB)
        if not is_depth:
            color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            image = color_jitter(image)

    # 5. Convert to Tensor
    return F_vision.to_tensor(image)


def custom_decoder(data):
    main_image_bytes = io.BytesIO(data['rgb'])
    depth_image_bytes = np.frombuffer(data['depth'], dtype=np.uint8)

    main_image = np.array(Image.open(main_image_bytes))
    depth_data = cv2.imdecode(depth_image_bytes, cv2.IMREAD_UNCHANGED)

    return main_image, depth_data


def merge(data):
    rgb, depth = data

    # rgb is (H, W, 3), depth is (H, W)
    rgb = cv2.resize(rgb, (448, 448))
    depth = cv2.resize(depth, (448, 448))

    # Convert to float32 and scale to [0, 1]
    rgb = rgb.astype(np.float32) / 255.0
    depth = depth.astype(np.float32) / 65535.0  # Assuming 16-bit depth

    return np.concatenate((
        rgb,
        np.expand_dims(depth, axis=2)
    ), axis=-1)


def to_tensor(batch):
    if isinstance(batch, list):
        # We expect a list of tensors from batched()
        return torch.stack(batch)
    return torch.as_tensor(batch)


def to_tuple(data):
    return (data,)


def load_dataset(config):
    # todo
    #  add jittering
    #  add to tensor
    train_dataset = (
        wds.WebDataset(train_path, resampled=True)
        .shuffle(10000)
        .map(custom_decoder)
        .map(merge)
        .map(F_vision.to_tensor)
        .batched(config.mini_batch_size, collation_fn=to_tensor)
    )

    test_dataset = (
        wds.WebDataset(test_path, resampled=True)
        .shuffle(10000)
        .map(custom_decoder)
        .map(merge)
        .map(F_vision.to_tensor)
        .batched(config.mini_batch_size, collation_fn=to_tensor)
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset = (
        wds.WebDataset("/Users/belle/Developer/belle-bot/belle_bot/vision/encoder/training/vision-encoder/v1.2/train-000000.tar", resampled=True)
        .shuffle(10000)
        .map(custom_decoder)
        # // todo jitter here
        .map(merge((448, 448)))
        # .map(jitter)
        # .to_tuple("rgb", "depth")
        # .map(preprocess_train_wrapper)
        # .batched(1000)
    )

    for merged in train_dataset:


        plt.subplot(1, 2, 1)
        plt.imshow(merged[:, :, :3])
        plt.subplot(1, 2, 2)
        plt.imshow(merged[:, :, 3])
        plt.show()
        # breakpoint()
