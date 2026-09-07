from pydantic import BaseModel, Field


class PositioningTrainingConfig(BaseModel):

    # Log a message to the cli output every 50k stepson the current window of training
    log_every_n_steps: int = 50_000

    # Save the model weights every 50_000 steps
    checkpoint_every_n_steps: int | None = 50_000

    train_every_n_steps: int | None = 32

    # max number of training steps
    max_steps: int = 500_000

    # The number of samplers per update
    mini_batch_size: int = 256

    # The max number of items stored in the replay buffer
    replay_buffer_size: int = 10000

    # The optimiser learning rate
    learning_rate: float = 1e-4

    # The rate to which the learning rate decays. lr = lr * (1- (step/max_steps)^gamma)
    learning_rate_gamma: float = 0.4

    gaussian_noise_factor: float = 0.3

    # The number of parallel environments the agent trains within
    n_environments: int = 1

    # If true, the full window will not be used, only a subsequence of the replay
    random_subsample: bool = True

    # If true, the map will be rotated around randomly
    random_rotation: bool = True

    max_gps_snap_distance: float = Field(5, description="If the distance error reaches this value the model should snap to the correct position")
