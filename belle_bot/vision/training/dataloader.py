import cv2
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from belle_bot.vision.tasks.encoder.data_loader import colour_jitter, extract_encoder_input, merge
from belle_bot.vision.tasks.utils.data import get_infinite_dataloader


def extract_decoder_output(data, apply_depth_mask=True):
    # todo add position jitter to everything
    # todo occasionally mask out entire depth input
    # todo, if a task isn't being trained don't do the processing for it

    input = extract_encoder_input(data)
    x_rgb_t, x_depth_t = colour_jitter(input)
    y_rgb_t, y_depth_t = x_rgb_t.copy(), x_depth_t.copy()

    # Depth masking in decoder
    if 'depth_mask' in data and apply_depth_mask:
        depth_mask = np.frombuffer(data['depth_mask'], dtype=np.uint8).reshape((448, 448))
        depth_mask = jitter_depth_mask(depth_mask)
        x_depth_t[np.where(depth_mask == 0)] = 0  # mask out the input

    # Bounding boxes
    bounding_boxes = np.frombuffer(data['bounding_boxes'], dtype=np.float32).reshape((300, 6))

    # Segmentation mask
    segmentation = np.frombuffer(data['segmentation_mask'], dtype=np.uint8)
    segmentation = segmentation.reshape((480, 640))
    segmentation = cv2.resize(segmentation, (448, 448))
    segmentation = np.array(segmentation, dtype=np.long)

    return {
        "input": merge((x_rgb_t, x_depth_t)),
        "decoder": merge((y_rgb_t, y_depth_t)),
        "bounding_boxes": bounding_boxes,
        "segmentation": segmentation,
    }


def to_tensor(samples):
    joint = {}

    for sample in samples:
        for key, value in sample.items():
            if key not in joint: joint[key] = []
            joint[key].append(value)

    # Format for torch training
    joint['input'] = torch.from_numpy(np.array(joint['input'])).permute(0, 3, 1, 2)
    joint['decoder'] = torch.from_numpy(np.array(joint['decoder'])).permute(0, 3, 1, 2)
    joint['bounding_boxes'] = torch.from_numpy(np.array(joint['bounding_boxes']))
    joint['segmentation'] = torch.from_numpy(np.array(joint['segmentation']))

    return joint


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


def extract_decoder_output_train(data):
    # todo process this into dicts for everything to read
    return extract_decoder_output(data, apply_depth_mask=True)


def extract_decoder_output_eval(data):
    return extract_decoder_output(data, apply_depth_mask=False)


def load_dataset(config: VisionEncoderTrainingConfig, device):
    train_dataset = (
        wds.WebDataset(config.training_path, resampled=True)
        .shuffle(10000)
        .map(extract_decoder_output_train)
        .batched(config.mini_batch_size, collation_fn=to_tensor)
    )

    test_dataset = (
        wds.WebDataset(config.testing_path, resampled=True)
        .shuffle(10000)
        .map(extract_decoder_output_eval)
        .batched(config.mini_batch_size, collation_fn=to_tensor)
    )

    # Convert to pytorch dataloader
    pin_memory = device.type != 'mps'
    train_dataset = get_infinite_dataloader(
        DataLoader(train_dataset, batch_size=None, pin_memory=pin_memory))
    test_dataset = get_infinite_dataloader(
        DataLoader(test_dataset, batch_size=None, pin_memory=pin_memory))

    return train_dataset, test_dataset
