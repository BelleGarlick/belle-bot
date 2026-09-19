import base64
import io
from pathlib import Path

import webdataset as wds
from PIL import Image


import torchvision.transforms as T


# Dataset setup
base_path = Path(__file__).parent / "vision-encoder" / "v1" / "rgb"
train_path = str(base_path / "train-{000000..000045}.tar")
test_path = str(base_path / "test-{000000..000016}.tar")


def preprocess(image, device, jitter=False):
    if jitter:
        # Standard augmentation pipeline
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.ToTensor(),
        ])
    else:
        # Just resize and convert to tensor
        transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])

    return transforms(image).to(device)


def custom_decoder(key, data):
    if key.endswith(".jpg"):
        try:
            # Try raw binary first, then base64
            try:
                return Image.open(io.BytesIO(data))
            except Exception:
                return Image.open(io.BytesIO(base64.b64decode(data)))
        except Exception:
            return None
    return data

def load_dataset(config, devjce):
    train_dataset = (
        wds.WebDataset(train_path, resampled=True)
        .shuffle(10000)
        .decode(custom_decoder)
        .to_tuple("jpg")
        .map_tuple(lambda x: preprocess(x, devjce, jitter=True))
        .batched(config.training.mini_batch_size)
    )

    test_dataset = (
        wds.WebDataset(test_path, resampled=True)
        .shuffle(1000)
        .decode(custom_decoder)
        .to_tuple("jpg")
        .map_tuple(lambda x: preprocess(x, devjce, jitter=False))
        .batched(config.training.mini_batch_size)
    )

    return train_dataset, test_dataset
