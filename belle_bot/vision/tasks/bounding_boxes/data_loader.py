import numpy as np
import torch
import webdataset as wds

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from belle_bot.vision.tasks.encoder.data_loader import colour_jitter, extract_encoder_input, merge


def extract_decoder_output(data, apply_depth_mask=True):
    input = extract_encoder_input(data)
    input = colour_jitter(input)
    x = merge(input)

    y = np.frombuffer(data['bounding_boxes'], dtype=np.float32).reshape((300, 6))

    return x, y


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


def extract_decoder_output_train(data):
    return extract_decoder_output(data, apply_depth_mask=True)


def extract_decoder_output_eval(data):
    return extract_decoder_output(data, apply_depth_mask=False)


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
