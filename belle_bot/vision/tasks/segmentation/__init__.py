import torch
from torch.utils.data import DataLoader

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from belle_bot.vision.tasks.segmentation.data_loader import load_dataset as segmentation_dataset
from belle_bot.vision.tasks.utils.data import get_infinite_dataloader
from . import model, loss, train
from ..utils.models import TaskDefinition


def get_task_definition(config: VisionEncoderTrainingConfig, device: torch.device):
    """
    This function is used to get task definition for training the bounding box
    detection task of the visual cortex

    :param config: The training config
    :param device: The device being used
    :return: The decoder config
    """
    mdl = model.VisualCortexSegmentationDecoder(latent_dim=config.model.embedding_size).to(device)

    # Load checkpoint and freeze model
    if config.model.detection.file_path:
        mdl.load_state_dict(torch.load(config.model.detection.file_path, map_location=device))

    return TaskDefinition(
        model=mdl,
        perform_step=train.perform_step
    )
