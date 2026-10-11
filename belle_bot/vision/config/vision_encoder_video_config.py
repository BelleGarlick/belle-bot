from pydantic import BaseModel

from belle_bot.vision.config.vision_encoder_model_config import VisionEncoderModelConfig
from houston.client.py.config import HoustonConfig


class VisionEncoderVideoConfig(BaseModel):

    model: VisionEncoderModelConfig = VisionEncoderModelConfig()

    houston: HoustonConfig = HoustonConfig()

    model_path: str = "model-270000.pt"

    # video_path: str
