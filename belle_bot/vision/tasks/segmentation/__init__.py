import torch
from torch.utils.data import DataLoader

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from belle_bot.vision.tasks.segmentation.data_loader import load_dataset as segmentation_dataset
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
    mdl = model.VisualCortexDecoder(latent_dim=config.model.embedding_size).to(device)

    # Load checkpoint and freeze model
    if config.model.detection.file_path:
        mdl.load_state_dict(torch.load(config.model.detection.file_path, map_location=device))

    # Prepare the train/test datasets
    pin_memory = device.type != 'mps'
    segmentation_train_dataset, segmentation_test_dataset = segmentation_dataset(config)
    segmentation_train_loader = iter(DataLoader(segmentation_train_dataset, batch_size=None, pin_memory=pin_memory))
    segmentation_test_loader = iter(DataLoader(segmentation_test_dataset, batch_size=None, pin_memory=pin_memory))

    return TaskDefinition(
        model=mdl,
        perform_step=train.perform_step,
        train_set=segmentation_train_loader,
        test_set=segmentation_test_loader
    )
