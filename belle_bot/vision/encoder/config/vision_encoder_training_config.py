from pydantic import BaseModel, Field


class VisionEncoderTrainingConfig(BaseModel):

    # # Log a message to the cli output every 50k steps on the current window of training
    # log_every_n_steps: int = Field(50_000, description="Log a message to the cli output every 50k steps on the current window of training")
    #
    # # Save the model weights every 50_000 steps
    # checkpoint_every_n_steps: int | None = Field(50_000, description="Save the model weights every 50_000 steps")

    # Eval the model weights every 50_000 steps
    eval_every_n_steps: int | None = Field(10_000, description="Eval the model weights every 250_000 steps")

    # train_every_n_steps: int | None = 24

    # max number of training steps
    max_steps: int = Field(100_000, description="max number of training steps")

    # The number of samplers per update
    mini_batch_size: int = Field(8, description="The number of samplers per update")

    # The optimiser learning rate
    learning_rate: float = Field(8e-4, description="The optimiser learning rate")

    # The rate to which the learning rate decays. lr = lr * (1- (step/max_steps)^gamma)
    learning_rate_gamma: float = Field(0.4, description="The rate to which the learning rate decays. lr = lr * (1- (step/max_steps)^gamma)")

    kl_annealing: float = 0.1
