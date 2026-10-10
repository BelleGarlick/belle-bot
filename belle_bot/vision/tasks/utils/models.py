from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from belle_bot.vision.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from belle_bot.vision.tasks.encoder.model import VisualCortexEncoder


@dataclass
class TaskDefinition:

    model: nn.Module

    perform_step: Callable[[VisualCortexStepArgs], dict[str, torch.Tensor]]

    train_set: iter

    test_set: iter


@dataclass
class VisualCortexStepArgs:

    config: VisionEncoderTrainingConfig

    batch: Any

    vision_encoder: VisualCortexEncoder

    model: nn.Module

    device: torch.device

    percentage_complete: float
