from pydantic import BaseModel

from belle_bot.houston.client.py.config import HoustonConfig
from belle_bot.mapping.positioning.config.positioning_model_config import PositioningModelConfig
from belle_bot.mapping.positioning.config.positioning_training_config import PositioningTrainingConfig


class MlFlowConfig(BaseModel):

    endpoint: str = "http://192.168.0.182:5000"


class PositioningConfig(BaseModel):

    training: PositioningTrainingConfig = PositioningTrainingConfig()

    model: PositioningModelConfig = PositioningModelConfig()

    mlflow: MlFlowConfig = MlFlowConfig()

    houston: HoustonConfig = HoustonConfig()
