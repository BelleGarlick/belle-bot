from pydantic import BaseModel, Field

from houston.client.py.config import HoustonConfig
from belle_bot.mapping.positioning.config.positioning_model_config import PositioningModelConfig
from belle_bot.mapping.positioning.config.positioning_training_config import PositioningTrainingConfig


class MlFlowConfig(BaseModel):

    enabled: bool = Field(True, description="Enables Houston training")

    endpoint: str = "http://houston:5000"

    experiment_name: str = Field("Visual Cortex Run", description="Ml Flow experiment name")


class PositioningConfig(BaseModel):

    training: PositioningTrainingConfig = PositioningTrainingConfig()

    model: PositioningModelConfig = PositioningModelConfig()

    mlflow: MlFlowConfig = MlFlowConfig()

    houston: HoustonConfig = HoustonConfig()
