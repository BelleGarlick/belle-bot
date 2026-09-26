from pydantic import BaseModel, Field

from belle_bot.mapping.positioning.config.positioning_config import MlFlowConfig
from belle_bot.vision.encoder.config.vision_encoder_model_config import VisionEncoderModelConfig
from houston.client.py.config import HoustonConfig


class VisionEncoderTrainingConfig(BaseModel):

    model: VisionEncoderModelConfig = VisionEncoderModelConfig()

    mlflow: MlFlowConfig = MlFlowConfig()

    houston: HoustonConfig = HoustonConfig()

    # Save the model weights every 50_000 steps
    checkpoint_every_n_steps: int | None = Field(20_000, description="Save the model weights every 50_000 steps")

    # Eval the model weights every 50_000 steps
    eval_every_n_steps: int | None = Field(10_000, description="Eval the model weights every 250_000 steps")

    # max number of training steps
    max_steps: int = Field(200_000, description="max number of training steps")

    # The number of samplers per update
    mini_batch_size: int = Field(16, description="The number of samplers per update")

    # The optimiser learning rate
    learning_rate: float = Field(1e-3, description="The optimiser learning rate")

    # The rate to which the learning rate decays. lr = lr * (1- (step/max_steps)^gamma)
    learning_rate_gamma: float = Field(0.4, description="The rate to which the learning rate decays. lr = lr * (1- (step/max_steps)^gamma)")

    kl_annealing: float = 0.0005
