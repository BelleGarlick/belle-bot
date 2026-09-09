import math
import os.path
from pathlib import Path
import random
import uuid
from collections import deque

import numpy as np

import torch
import torch.nn.functional as F
import mlflow

from belle_bot.mapping.positioning.config.positioning_config import PositioningConfig
from belle_bot.mapping.positioning.training.environment.env import Frame
from belle_bot.mapping.positioning.training.environment.episode_processor import load_episodes
from belle_bot.mapping.positioning.training.environment.multi_environment import MultiEnvironment
from belle_bot.mapping.positioning.training.environment.preprocessor import process_state
from belle_bot.mapping.positioning.training.ml_model import PositionalModelling
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds
from belle_bot.mapping.positioning.training.testing import perform_evals
from belle_bot.mapping.positioning.training.utils import ReplayBuffer, TrainingSample
from belle_bot.utils.cli import clpy

# todo
#  estimate variance when training
#  drop sections randomly
#  create normalisation from some initial steps
#  rotate the scene
#  add camera
#  bspline gps
#  cli args to trigger training runs
#  add testing set here

os.environ["CACHE_REPLAYS"] = "true"

config = clpy.parse_cli_args(PositioningConfig())

config.training.checkpoint_every_n_steps = None

INITIAL_TRAIN_SIZE = 500  # used to accumulate data for normalisation
RANDOM_SEED = 42


EXPERIMENT_TAG = "all 7"
# for 8
#   add snapping into eval so eval results are better


# instead, sample more items, but only train on the items where the error is larger. so it becomes a sort of heirstic search. doing so means we're not wasting cycles train pointeless data.

device = torch.device('cuda')


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


def train_model(buffer: ReplayBuffer):
    model.train()
    optimizer.zero_grad()

    batch_idxs = buffer.sample(config.training.mini_batch_size)

    modality_frames, modality_types, ys = sample(buffer, batch_idxs)
    prediction = model(modality_frames, modality_types)

    loss_per_sample = F.huber_loss(prediction, ys, reduction='none').mean(dim=-1)
    loss = loss_per_sample.mean()
    loss.backward()
    optimizer.step()

    buffer[batch_idxs] = loss_per_sample.detach().cpu().numpy()

    model.eval()
    return loss.item(), np.abs(modality_frames.detach().cpu().numpy()).max()


def validate_same_tag():
    if os.path.exists("train-tag.txt"):
        with open("train-tag.txt", "r") as f:
            if f.readline().strip() != EXPERIMENT_TAG:
                raise ValueError("Changed train tag")
    else:
        with open("train-tag.txt", "w") as f:
            f.write(EXPERIMENT_TAG)


if __name__ == "__main__":
    # todo write a new way to create normalisation bounds. currently we have no way to fit the bounds
    mlflow.set_tracking_uri(config.mlflow.endpoint)

    bounds_path = Path(__file__).parent / "bounds.json"
    bounds = NormalisationBounds().load(bounds_path)

    for _ in range(100):
        config.training.actual_snap_distance = random.randint(3, 7)
        config.training.gaussian_noise_factor = random.random() * 0.1
        config.training.max_gps_snap_distance = random.randint(3, 7)
        config.model.embedding_size = random.choice([32, 48, 64, 96, 128])
        config.training.train_every_n_steps = random.choice([16, 20, 24, 28, 32, 36])
        config.model.sequence_length = random.randint(100, 200)
        config.training.replay_buffer_size = random.randint(1000, 50_000)
        config.training.learning_rate_gamma = (random.random() * 0.4) + 0.2
        config.training.learning_rate = random.choice([5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3])
        config.training.mini_batch_size = random.randint(8, 384)
        config.training.n_environments = random.randint(1, 10)

        clpy.print_values(config)

        model = PositionalModelling(13, config.model, out_scale=config.training.max_gps_snap_distance).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

        env = MultiEnvironment(
            config,
            subset="training",
            envs=config.training.n_environments,
            seq_len=config.model.sequence_length,
            random_subsample=config.training.random_subsample,
            random_rotation=config.training.random_rotation,
            seed=RANDOM_SEED
        )

        states: list[list[Frame]] = env.reset().initial_states
        # Start sample from the environment
        buffer = ReplayBuffer(config.training.replay_buffer_size)
        episode_step_error = []
        episode_losses = []
        magnitudes = deque(maxlen=100)

        mlflow.set_experiment("positioning")

        with mlflow.start_run(run_name=str(uuid.uuid4())):
            mlflow.log_params(clpy.to_dict(config))
            mlflow.set_tag("experiment", EXPERIMENT_TAG)

            model.eval()
            for step in range(0, config.training.max_steps):
                env_id: int = step % len(env)

                percentage_complete = step / config.training.max_steps
                percentage_remaining = 1 - percentage_complete

                # Scale the learning rate with the percentage_complete
                scale = 1 - math.pow(percentage_complete, config.training.learning_rate_gamma)
                for g in optimizer.param_groups:
                    g['lr'] = config.training.learning_rate * scale

                # Process and create the model input to what the target change should be then predict the position for it
                modality_data, modality_types = process_state(states[env_id], seq_length=config.model.sequence_length, normalisation_bounds=bounds)
                with torch.no_grad():
                    predicted_position_change = model(
                        torch.tensor(modality_data, device=device, dtype=torch.float32),
                        torch.tensor(modality_types, device=device, dtype=torch.int64),
                    )
                    predicted_position_change = predicted_position_change.cpu().numpy()[0]

                # Calculate noise which is added to the step
                position_noise_factor = percentage_remaining * config.training.gaussian_noise_factor

                # Use a deterministic seed for noise based on the current step and RANDOM_SEED
                noise_rng = np.random.default_rng(RANDOM_SEED + step)
                position_noise = noise_rng.uniform(-1, 1, predicted_position_change.shape).astype(np.float32) * position_noise_factor

                # Perform the step change
                # THE SNAPPING IS NOT DEFINED HERE??? Need to try enabling it. atm the error is just done on it's own?
                new_state, terminated = env.step(
                    env_id,
                    predicted_position_change + position_noise,
                    max_error=config.training.actual_snap_distance
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

                # if len(buffer) == INITIAL_TRAIN_SIZE:
                #     train_normaliser(buffer)

                mini_batch_loss = None
                if len(buffer) >= config.training.mini_batch_size and (step + 1) % config.training.train_every_n_steps == 0:
                    mini_batch_loss, mb_magnitude = train_model(buffer)
                    episode_losses.append(mini_batch_loss)
                    magnitudes.append(mb_magnitude)

                # Reset env if terminated
                if terminated:
                    states[env_id] = env.reset(env_id).initial_states[0]

                # Print a status update every x steps
                if (step + 1) % config.training.log_every_n_steps == 0:
                    mean_step_err = np.mean(episode_step_error)
                    mean_loss = np.mean(episode_losses) if episode_losses else 0.0

                    episode_step_error = []
                    episode_losses = []
                    if config.training.checkpoint_every_n_steps is not None \
                            and(step + 1) % config.training.checkpoint_every_n_steps == 0:
                        model_path = f"model-{step + 1}.pt"
                        torch.save(model.state_dict(), model_path)
                        mlflow.log_artifact(model_path)

                    eval = perform_evals(
                        episodes=load_episodes(config, "testing"),
                        model=model,
                        bounds=bounds,
                    )

                    mlflow.log_metrics({
                        "mean_loss_window": mean_loss,
                        "mean_step_error_window": mean_step_err,
                        "step": step + 1
                    }, step=step)

                    print("\r{} Mean Step {:.5f} Loss {:.5f} MB mag: {:.5f}".format(
                        step + 1,
                        mean_step_err,
                        mean_loss,
                        np.mean(magnitudes)
                    ))

                elif step % 50 == 0:
                    print("\r{} Mean Step {:.5f} Loss {:.5f} MB mag: {:.5f}".format(
                        step,
                        np.mean(episode_step_error),
                        np.mean(episode_losses) if episode_losses else 0.0,
                        np.mean(magnitudes) if magnitudes else 0.0
                        # buffer.mean_loss()
                    ), end="")

        eval = perform_evals(
            config=config,
            episodes=load_episodes(config, "testing"),
            model=model,
            bounds=bounds,
        )

        mlflow.log_metrics({
            "mean_position_error": eval["mean_position_error"],
            "mean_final_position_error": eval["mean_final_position_error"],
        }, step=step)


        validate_same_tag()
