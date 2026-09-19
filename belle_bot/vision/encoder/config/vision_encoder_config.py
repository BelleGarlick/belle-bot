from pydantic import BaseModel

from belle_bot.mapping.positioning.config.positioning_config import MlFlowConfig
from belle_bot.vision.encoder.config.vision_encoder_model_config import VisionEncoderModelConfig
from belle_bot.vision.encoder.config.vision_encoder_training_config import VisionEncoderTrainingConfig
from houston.client.py.config import HoustonConfig


class VisionEncoderConfig(BaseModel):

    training: VisionEncoderTrainingConfig = VisionEncoderTrainingConfig()

    model: VisionEncoderModelConfig = VisionEncoderModelConfig()

    mlflow: MlFlowConfig = MlFlowConfig()

    houston: HoustonConfig = HoustonConfig()
