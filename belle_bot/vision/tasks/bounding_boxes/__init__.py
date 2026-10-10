import torch
from torch.utils.data import DataLoader

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from . import train, utils, model, data_loader
from .utils import CLASSES
from ..utils.models import TaskDefinition
from belle_bot.vision.tasks.bounding_boxes.data_loader import load_dataset as detection_dataset


def get_task_definition(config: VisionEncoderTrainingConfig, device: torch.device):
    """
    This function is used to get task definition for training the bounding box
    detection task of the visual cortex

    :param config: The training config
    :param device: The device being used
    :return: The decoder config
    """
    mdl = model.VisualCortexBoundingBoxModel(
        latent_dim=config.model.embedding_size,
        num_classes=len(CLASSES)
    ).to(device)

    # Load checkpoint and freeze model
    if config.model.detection.file_path:
        mdl.load_state_dict(torch.load(config.model.detection.file_path, map_location=device))
    if config.model.detection.frozen:
        for param in mdl.parameters():
            param.requires_grad = False

    # Prepare the train/test datasets
    pin_memory = device.type != 'mps'
    detection_train_dataset, detection_test_dataset = detection_dataset(config)
    detection_train_loader = iter(DataLoader(detection_train_dataset, batch_size=None, pin_memory=pin_memory))
    detection_test_loader = iter(DataLoader(detection_test_dataset, batch_size=None, pin_memory=pin_memory))

    return TaskDefinition(
        model=mdl,
        perform_step=train.perform_step,
        train_set=detection_train_loader,
        test_set=detection_test_loader
    )
