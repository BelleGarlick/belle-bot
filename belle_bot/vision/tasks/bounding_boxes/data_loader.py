import numpy as np
import torch
import webdataset as wds

from belle_bot.vision.tasks.encoder.data_loader import colour_jitter, extract_encoder_input, merge

# Dataset setup
train_path = str("/Users/belle/Developer/belle-bot/belle_bot/vision/encoder/dataset/distillation-test-000000.tar")
test_path = str("/Users/belle/Developer/belle-bot/belle_bot/vision/encoder/dataset/distillation-test-000000.tar")


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


def load_dataset(config):
    train_dataset = (
        wds.WebDataset(train_path, resampled=True)
        .shuffle(10000)
        .map(extract_decoder_output_train)
        .batched(config.mini_batch_size, collation_fn=to_tensor)
    )

    test_dataset = (
        wds.WebDataset(test_path, resampled=True)
        .shuffle(10000)
        .map(extract_decoder_output_eval)
        .batched(config.mini_batch_size, collation_fn=to_tensor)
    )

    return train_dataset, test_dataset
