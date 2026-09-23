import base64
import io
from pathlib import Path

import webdataset as wds
from PIL import Image


import torchvision.transforms as T


# Dataset setup
base_path = Path(__file__).parent / "vision-encoder" / "v1.1"
train_path = str(base_path / "train-{000000..000000}.tar")
test_path = str(base_path / "test-{000000..000000}.tar")


def preprocess(image, jitter=False, is_depth=False):
    import torchvision.transforms.functional as F_vision
    import random

    # 1. Resize
    image = F_vision.resize(image, (224, 224))

    if jitter:
        # 2. Random Resized Crop
        # Note: We need to use the same parameters for both images in a pair.
        # This function should probably be called with params already calculated if we want perfect sync.
        # But for now, we'll try to use the same random seed in preprocess_train.
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.8, 1.0), ratio=(0.75, 1.33))
        image = F_vision.resized_crop(image, i, j, h, w, (224, 224))

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
    # This decoder will be applied to EACH value in the sample dict
    if isinstance(data, bytes):
        try:
            return Image.open(io.BytesIO(data))
        except Exception:
            try:
                return Image.open(io.BytesIO(base64.b64decode(data)))
            except Exception:
                return data
    return data

def preprocess_train(rgb, depth):
    # Use the same seed/random state for both to ensure identical crops/flips
    import torch
    import random
    
    if not isinstance(rgb, Image.Image) or not isinstance(depth, Image.Image):
        # Fallback if decoder didn't work as expected
        if isinstance(rgb, bytes): rgb = Image.open(io.BytesIO(rgb))
        if isinstance(depth, bytes): depth = Image.open(io.BytesIO(depth))

    state = torch.get_rng_state()
    py_state = random.getstate()
    
    rgb_t = preprocess(rgb, jitter=True, is_depth=False)
    
    torch.set_rng_state(state)
    random.setstate(py_state)
    depth_t = preprocess(depth, jitter=True, is_depth=True)
    
    # rgb_t is (3, 224, 224), depth_t is (3, 224, 224) if it's RGB encoded, but we only want one channel from depth
    if depth_t.shape[0] == 3:
        depth_t = depth_t[0:1, :, :]
        
    return torch.cat([rgb_t, depth_t], dim=0)

def preprocess_test(rgb, depth):
    import torch
    if not isinstance(rgb, Image.Image) or not isinstance(depth, Image.Image):
        if isinstance(rgb, bytes): rgb = Image.open(io.BytesIO(rgb))
        if isinstance(depth, bytes): depth = Image.open(io.BytesIO(depth))
    rgb_t = preprocess(rgb, jitter=False)
    depth_t = preprocess(depth, jitter=False)

    if depth_t.shape[0] == 3:
        depth_t = depth_t[0:1, :, :]

    return torch.cat([rgb_t, depth_t], dim=0)

def preprocess_train_wrapper(x):
    return (preprocess_train(*x),)

def preprocess_test_wrapper(x):
    return (preprocess_test(*x),)

def load_dataset(config):
    train_dataset = (
        wds.WebDataset(train_path, resampled=True)
        .shuffle(10000)
        .map(custom_decoder)
        .to_tuple("rgb", "depth")
        .map(preprocess_train_wrapper)
        .batched(config.mini_batch_size)
    )

    test_dataset = (
        wds.WebDataset(test_path, resampled=True)
        .shuffle(10000)
        .map(custom_decoder)
        .to_tuple("rgb", "depth")
        .map(preprocess_test_wrapper)
        .batched(config.mini_batch_size)
    )

    return train_dataset, test_dataset
