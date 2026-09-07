from pydantic import BaseModel

from belle_bot.mapping.positioning.config.positioning_model_config import PositioningModelConfig
from belle_bot.mapping.positioning.config.positioning_training_config import PositioningTrainingConfig


class PositioningConfig(BaseModel):

    training: PositioningTrainingConfig = PositioningTrainingConfig()

    model: PositioningModelConfig = PositioningModelConfig()
