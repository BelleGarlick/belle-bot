from pydantic import BaseModel, Field


class PositioningTrainingConfig(BaseModel):

    # Log a message to the cli output every 50k steps on the current window of training
    log_every_n_steps: int = Field(50_000, description="Log a message to the cli output every 50k steps on the current window of training")

    # Save the model weights every 50_000 steps
    checkpoint_every_n_steps: int | None = Field(50_000, description="Save the model weights every 50_000 steps")

    train_every_n_steps: int | None = 32

    # max number of training steps
    max_steps: int = Field(750_000, description="max number of training steps")

    # The number of samplers per update
    mini_batch_size: int = Field(256, description="The number of samplers per update")

    # The max number of items stored in the replay buffer
    replay_buffer_size: int = Field(10000, description="The max number of items stored in the replay buffer")

    # The optimiser learning rate
    learning_rate: float = Field(1e-4, description="The optimiser learning rate")

    # The rate to which the learning rate decays. lr = lr * (1- (step/max_steps)^gamma)
    learning_rate_gamma: float = Field(0.4, description="The rate to which the learning rate decays. lr = lr * (1- (step/max_steps)^gamma)")

    gaussian_noise_factor: float = 0.3

    # The number of parallel environments the agent trains within
    n_environments: int = Field(1, description="The number of parallel environments the agent trains within")

    # If true, the full window will not be used, only a subsequence of the replay
    random_subsample: bool = Field(True, description="If true, the full window will not be used, only a subsequence of the replay")

    # If true, the map will be rotated around randomly
    random_rotation: bool = Field(True, description="If true, the map will be rotated around randomly")

    max_gps_snap_distance: float = Field(5, description="If the distance error reaches this value the model should snap to the correct position (scale?)")

    actual_snap_distance: float = Field(5, description="If the distance error reaches this value the model should snap to the correct position")
