import math
import random
from collections import deque

import numpy as np

import torch
import torch.nn.functional as F
import mlflow

from belle_bot.mapping.positioning.training.environment.env import Frame
from belle_bot.mapping.positioning.training.environment.multi_environment import MultiEnvironment
from belle_bot.mapping.positioning.training.environment.preprocessor import process_state
from belle_bot.mapping.positioning.training.ml_model import PositionalModelling
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds
from belle_bot.mapping.positioning.training.utils import ReplayBuffer, TrainingSample
from belle_bot.mapping.positioning.training.seeding import set_seed

# todo
#  estimate variance when training
#  drop sections randomly
#  create normalisation from some initial steps
#  rotate the scene
#  add camera
#  bspline gps
#  cli args to trigger training runs
#  add testing set here


LOG_EVERY_N_STEPS = 50_000
SAVE_EVERY_N_STEPS = 50_000
MAX_STEPS = 500_000
SEQUENCE_LENGTH = 100
N_LAYERS = 2

TRAIN_EVERY_N_STEPS = 8
MINI_BATCH_SIZE = 256
MAX_SNAP_GPS_DISTANCE = 10
INITIAL_TRAIN_SIZE = 500  # used to accumulate data for normalisation
REPLAY_BUFFER_SIZE = 10000
N_ENVIRONMENTS = 10
GAUSSIAN_NOISE_FACTOR = 0.0
LEARNING_RATE = 1e-3
LR_GAMMA = 0.3
RANDOM_SEED = 42


# instead, sample more items, but only train on the items where the error is larger. so it becomes a sort of heirstic search. doing so means we're not wasting cycles train pointeless data.

device = torch.device('mps')


# todo also make make it so that eventually we can make something more stochastic where frames are dropped / not perfectly discrete.
# todo update the testing
#  - add rotated maps. need to handle magnetometer for that tho



def sample(buffer: ReplayBuffer, idxs: list[int]):
    modality_frames, modality_types, ys = [], [], []
    for idx in idxs:
        sample = buffer[idx]
        modality_frames.append(sample.model_input[0])
        modality_types.append(sample.model_input[1])
        ys.append(sample.target)

    return (
        torch.tensor(np.concatenate(modality_frames), dtype=torch.float32, device=device),
        torch.tensor(np.concatenate(modality_types), dtype=torch.int64, device=device),
        torch.tensor(np.array(ys), dtype=torch.float32, device=device),
    )


def train_model(buffer: ReplayBuffer, step: int):
    model.train()
    optimizer.zero_grad()

    # Use a deterministic seed for sampling based on the current step and RANDOM_SEED
    sample_seed = RANDOM_SEED + step
    batch_idxs = buffer.sample(MINI_BATCH_SIZE, seed=sample_seed)

    modality_frames, modality_types, ys = sample(buffer, batch_idxs)
    prediction = model(modality_frames, modality_types)

    loss_per_sample = F.huber_loss(prediction, ys, reduction='none').mean(dim=-1)
    loss = loss_per_sample.mean()
    loss.backward()
    optimizer.step()

    buffer[batch_idxs] = loss_per_sample.detach().cpu().numpy()

    model.eval()
    return loss.item(), np.abs(modality_frames.detach().cpu().numpy()).max()


def train_normaliser(buffer: ReplayBuffer):
    pass
    # bounds.fit(buffer).save("bounds.json")


if __name__ == "__main__":
    # todo write a new way to create normalisation bounds. currently we have no way to fit the bounds
    bounds = NormalisationBounds().load("bounds.json")

    for _ in range(100):
        EMBEDDING_SIZE = int(math.pow(random.random(), 1.5) * 512)

        set_seed(RANDOM_SEED)

        model = PositionalModelling(13, EMBEDDING_SIZE, n_layers=N_LAYERS, out_scale=MAX_SNAP_GPS_DISTANCE).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        # todo add rotation augmentation bool
        env = MultiEnvironment(
            subset="training",
            envs=N_ENVIRONMENTS,
            seq_len=SEQUENCE_LENGTH,
            random_subsample=True,
            random_rotation=True,
            seed=RANDOM_SEED
        )

        states: list[list[Frame]] = env.reset().initial_states
        # Start sample from the environment
        buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        episode_step_error = []
        episode_losses = []
        magnitudes = deque(maxlen=100)

        mlflow.set_experiment("positioning")

        with mlflow.start_run(run_name="Layers {}x{}, MB {} RBS {}".format(N_LAYERS, EMBEDDING_SIZE, MINI_BATCH_SIZE, REPLAY_BUFFER_SIZE)):
            # Re-set seed inside the MLflow run to ensure all environment setups and data loading are deterministic
            set_seed(RANDOM_SEED)

            mlflow.log_params({
                "sequence_length": SEQUENCE_LENGTH,
                "mini_batch_size": MINI_BATCH_SIZE,
                "max_steps": MAX_STEPS,
                "learning_rate": LEARNING_RATE,
                "learning_rate_gamma": LR_GAMMA,
                "replay_buffer_size": REPLAY_BUFFER_SIZE,
                "n_environments": N_ENVIRONMENTS,
                "gaussian_noise_factor": GAUSSIAN_NOISE_FACTOR,
                "embedding_size": EMBEDDING_SIZE,
                "n_layers": N_LAYERS,
                "train_every_n_steps": TRAIN_EVERY_N_STEPS,
                "random_seed": RANDOM_SEED,
            })
            mlflow.set_tag("experiment", "model size")

            model.eval()
            for step in range(0, MAX_STEPS):
                env_id: int = step % len(env)

                percentage_complete = step / MAX_STEPS
                percentage_remaining = 1 - percentage_complete

                # Scale the learning rate with the percentage_complete
                scale = 1 - math.pow(step / MAX_STEPS, LR_GAMMA)
                for g in optimizer.param_groups:
                    g['lr'] = LEARNING_RATE * scale

                # Process and create the model input to what the target change should be then predict the position for it
                modality_data, modality_types = process_state(states[env_id], seq_length=SEQUENCE_LENGTH, normalisation_bounds=bounds)
                with torch.no_grad():
                    predicted_position_change = model(
                        torch.tensor(modality_data, device=device, dtype=torch.float32),
                        torch.tensor(modality_types, device=device, dtype=torch.int64),
                    )
                    predicted_position_change = predicted_position_change.cpu().numpy()[0]

                # Calculate noise which is added to the step
                position_noise_factor = percentage_remaining * GAUSSIAN_NOISE_FACTOR

                # Use a deterministic seed for noise based on the current step and RANDOM_SEED
                noise_rng = np.random.default_rng(RANDOM_SEED + step)
                position_noise = noise_rng.uniform(-1, 1, predicted_position_change.shape).astype(np.float32) * position_noise_factor

                # Perform the step change
                new_state, terminated = env.step(
                    env_id,
                    predicted_position_change + position_noise
                )

                true_position_change = states[env_id][-1].position_change

                step_error = np.linalg.norm(true_position_change - predicted_position_change)
                episode_step_error.append(step_error)

                buffer.append(
                    TrainingSample(
                        model_input=(modality_data, modality_types),
                        target=true_position_change
                    ),
                    error=F.huber_loss(
                        torch.tensor(true_position_change, device=device, dtype=torch.float32),
                        torch.tensor(predicted_position_change, device=device, dtype=torch.float32),
                    ).detach().cpu().item(),
                )

                states[env_id] = new_state

                if len(buffer) == INITIAL_TRAIN_SIZE:
                    train_normaliser(buffer)

                mini_batch_loss = None
                if len(buffer) >= MINI_BATCH_SIZE and (step + 1) % TRAIN_EVERY_N_STEPS == 0:
                    mini_batch_loss, mb_magnitude = train_model(buffer, step)
                    episode_losses.append(mini_batch_loss)
                    magnitudes.append(mb_magnitude)

                # Reset env if terminated
                if terminated:
                    states[env_id] = env.reset(env_id).initial_states[0]

                # Print a status update every x steps
                if (step + 1) % LOG_EVERY_N_STEPS == 0:
                    mean_step_err = np.mean(episode_step_error)
                    mean_loss = np.mean(episode_losses) if episode_losses else 0.0
                    print("\r{} Mean Step {:.5f} Loss {:.5f} MB mag: {:.5f}".format(
                        step + 1,
                        mean_step_err,
                        mean_loss,
                        np.mean(magnitudes)
                    ))
                    mlflow.log_metric("mean_step_error_window", mean_step_err, step=step)
                    mlflow.log_metric("mean_loss_window", mean_loss, step=step)
                    episode_step_error = []
                    episode_losses = []
                    # if (step + 1) % SAVE_EVERY_N_STEPS == 0:
                    #     model_path = f"model-{step + 1}.pt"
                        # torch.save(model.state_dict(), model_path)
                        # mlflow.log_artifact(model_path)

                elif step % 10 == 0:
                    print("\r{} Mean Step {:.5f} Loss {:.5f} MB mag: {:.5f}".format(
                        step,
                        np.mean(episode_step_error),
                        np.mean(episode_losses) if episode_losses else 0.0,
                        np.mean(magnitudes) if magnitudes else 0.0
                        # buffer.mean_loss()
                    ), end="")
