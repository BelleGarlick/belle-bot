import torch

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from . import model, loss, train
from ..utils.models import TaskDefinition
from belle_bot.vision.tasks.decoder.data_loader import load_dataset as decoder_dataset


def get_task_definition(config: VisionEncoderTrainingConfig, device: torch.device):
    """
    This function is used to get task definition for training the decoder task of the visual cortex

    :param config: The training config
    :param device: The device being used
    :return: The decoder config
    """
    mdl = model.VisualCortexDecoder().to(device)

    # Load checkpoint and freeze model
    if config.model.decoder.file_path:
        mdl.load_state_dict(torch.load(config.model.decoder.file_path, map_location=device))

    return TaskDefinition(
        model=mdl,
        perform_step=train.perform_step
    )
