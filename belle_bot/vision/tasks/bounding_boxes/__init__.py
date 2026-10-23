import torch

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

    return TaskDefinition(
        model=mdl,
        perform_step=train.perform_step
    )
