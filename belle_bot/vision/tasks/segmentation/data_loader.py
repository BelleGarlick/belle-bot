import cv2
import numpy as np
import torch
import webdataset as wds

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from belle_bot.vision.tasks.encoder.data_loader import extract_encoder_input, merge


def extract_segmentation_data(data, apply_depth_mask=True):
    input = extract_encoder_input(data)

    segmask = np.frombuffer(data['segmentation_mask'], dtype=np.uint8)
    segmask = segmask.reshape((480, 640))
    segmask = cv2.resize(segmask, (448, 448))
    segmask = np.array(segmask, dtype=np.long)

    return merge(input), segmask


def to_tensor(samples):
    x_batch = []
    y_batch = []
    for x, y in samples:
        # x, y are (H, W, C) -> (C, H, W)
        x_t = torch.from_numpy(x).permute(2, 0, 1)
        y_t = torch.from_numpy(y)
        x_batch.append(x_t)
        y_batch.append(y_t)
    return torch.stack(x_batch), torch.stack(y_batch)


def extract_segmentation_data_train(data):
    return extract_segmentation_data(data, apply_depth_mask=True)


def extract_segmentation_data_eval(data):
    return extract_segmentation_data(data, apply_depth_mask=False)


def load_dataset(config: VisionEncoderTrainingConfig):
    train_dataset = (
        wds.WebDataset(config.training_path, resampled=True)
        .shuffle(10000)
        .map(extract_segmentation_data_train)
        .batched(config.mini_batch_size, collation_fn=to_tensor)
    )

    test_dataset = (
        wds.WebDataset(config.testing_path, resampled=True)
        .shuffle(10000)
        .map(extract_segmentation_data_eval)
        .batched(config.mini_batch_size, collation_fn=to_tensor)
    )

    return train_dataset, test_dataset
