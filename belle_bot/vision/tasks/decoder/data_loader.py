import cv2
import numpy as np
import torch
import webdataset as wds

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from belle_bot.vision.tasks.encoder.data_loader import colour_jitter, extract_encoder_input, position_jitter, merge


def extract_decoder_ouput(data, apply_depth_mask=True):
    input = extract_encoder_input(data)
    input = colour_jitter(input)
    x_rgb_t, x_depth_t = position_jitter(input)
    y_rgb_t, y_depth_t = x_rgb_t.copy(), x_depth_t.copy()

    if 'depth_mask' in data and apply_depth_mask:
        depth_mask = np.frombuffer(data['depth_mask'], dtype=np.uint8).reshape((448, 448))
        depth_mask = jitter_depth_mask(depth_mask)
        x_depth_t[np.where(depth_mask == 0)] = 0  # mask out the input

    x = merge((x_rgb_t, x_depth_t))
    y = merge((y_rgb_t, y_depth_t))

    return x, y


def jitter_depth_mask(depth_mask):
    im_size = depth_mask.shape[0]

    # crop image randomly
    random_crop_size = np.random.randint(int(im_size * 0.9), im_size)
    random_x_offset = np.random.randint(0, im_size - random_crop_size)
    random_y_offset = np.random.randint(0, im_size - random_crop_size)

    # crop the image at the offsets and resize
    if depth_mask is not None:
        depth_mask = depth_mask[random_x_offset:random_x_offset + random_crop_size, random_y_offset:random_y_offset + random_crop_size]

    return cv2.resize(depth_mask, (448, 448))


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


def extract_decoder_output_train(data):
    return extract_decoder_ouput(data, apply_depth_mask=True)


def extract_decoder_output_eval(data):
    return extract_decoder_ouput(data, apply_depth_mask=False)


def load_dataset(config: VisionEncoderTrainingConfig):
    train_dataset = (
        wds.WebDataset(config.training_path, resampled=True)
        .shuffle(10000)
        .map(extract_decoder_output_train)
        .batched(config.micro_batch_size, collation_fn=to_tensor)
    )

    test_dataset = (
        wds.WebDataset(config.testing_path, resampled=True)
        .shuffle(10000)
        .map(extract_decoder_output_eval)
        .batched(config.micro_batch_size, collation_fn=to_tensor)
    )

    return train_dataset, test_dataset
