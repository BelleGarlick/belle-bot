from pydantic import BaseModel

from belle_bot.vision.encoder.config.vision_encoder_model_config import VisionEncoderModelConfig
from houston.client.py.config import HoustonConfig


class VisionEncoderTrainingConfig(BaseModel):

    model: VisionEncoderModelConfig = VisionEncoderModelConfig()

    houston: HoustonConfig = HoustonConfig()

    model_path: str = "model-270000.pt"

    # video_path: str
