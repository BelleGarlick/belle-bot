import torch
from . import model
from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig


def get_task_definition(config: VisionEncoderTrainingConfig, device: torch.device):
    from belle_bot.vision.tasks.utils.models import TaskDefinition

    mdl = model.VisualCortexEncoder().to(device)

    # Load checkpoint and freeze model
    if config.model.encoder.file_path:
        mdl.load_state_dict(torch.load(config.model.encoder.file_path, map_location=device))
    if config.model.encoder.frozen:
        for param in mdl.parameters():
            param.requires_grad = False

    # This task shouldn't be directly trained on. This is just here for completeness
    return TaskDefinition(
        model=mdl,
        perform_step=None
    )
