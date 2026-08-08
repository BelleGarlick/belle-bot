import math

import numpy as np

import torch
import torch.nn.functional as F
import mlflow

from belle_bot.mapping.positioning.training.environment.multi_environment import MultiEnvironment
from belle_bot.mapping.positioning.training.ml_model import PositionalModelling
from belle_bot.mapping.positioning.training.models import GpsPoint, ModalityEnum, ImuData
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds
from belle_bot.mapping.positioning.training.utils import ReplayBuffer, TrainingSample

# todo
#  estimate variance when training
#  drop sections randomly
#  create normalisation from some initial steps
#  rotate the scene
#  add camera
#  bspline gps
#  cli args to trigger training runs


SEQUENCE_LENGTH = 100
MINI_BATCH_SIZE = 512
LOG_EVERY_N_STEPS = 5000
SAVE_EVERY_N_STEPS = 20_000
MAX_SNAP_GPS_DISTANCE = 10
INITIAL_TRAIN_SIZE = 500  # used to accumulate data for normalisation
REPLAY_BUFFER_SIZE = 5000
N_ENVIRONMENTS = 4
GAUSSIAN_NOISE_FACTOR = 0.1
MAX_STEPS = 100_000
LEARNING_RATE = 1e-4
EMBEDDING_SIZE = 256
N_LAYERS = 2


# instead, sample more items, but only train on the items where the error is larger. so it becomes a sort of heirstic search. doing so means we're not wasting cycles train pointeless data.

device = torch.device('mps')


# todo also make make it so that eventually we can make something more stochastic where frames are dropped / not perfectly discrete.
# todo update the testing
#  - add rotated maps. need to handle magnetometer for that tho


# realistically, i think the best thing to do is to jus use frames which embed the diff item
def process_state(frames: list[ImuData | GpsPoint], seq_length, normalisation_bounds: NormalisationBounds):
    modality_types = [ModalityEnum.PAD] * seq_length
    modality_data = [[0] * 10 for _ in range(seq_length)]

    for item in frames:
        if isinstance(item, ImuData):
            # Update the modality data
            # todo change the angle to cos / sin so that it's measured a lil better at the point from -180 to 180
            modality_types.append(ModalityEnum.IMU)
            modality_data.append(np.hstack((
                item.acc / normalisation_bounds['imu.acc'],
                item.gyro / normalisation_bounds['imu.gyro'],
                item.angle / normalisation_bounds['imu.angle'],
                [item.timestamp]
            )))

        elif isinstance(item, GpsPoint):
            # Update the modality data
            modality_types.append(ModalityEnum.GPS)
            modality_data.append(np.array([
                item.x / normalisation_bounds["gps.x"],
                item.y / normalisation_bounds["gps.y"],
                item.altitude / normalisation_bounds["gps.alt"],
            ] + [0] * 6 + [item.timestamp]))  # padded the item so everything is same size

        else:
            # split up camera into multiple tokens
            raise NotImplementedError()

    modality_types = np.array([modality_types[-seq_length:]], dtype=np.int64)
    modality_data = np.array([modality_data[-seq_length:]], dtype=np.float32)

    return (
        modality_data,
        modality_types
    )


def sample(buffer: ReplayBuffer, idxs: list[int]):
    modality_frames, modality_types, ys = [], [], []
    for idx in idxs:
        modality_frames.append(buffer[idx].model_input[0])
        modality_types.append(buffer[idx].model_input[1])
        ys.append(buffer[idx].target)

    return (
        torch.tensor(np.concatenate(modality_frames), dtype=torch.float32, device=device),
        torch.tensor(np.concatenate(modality_types), dtype=torch.int64, device=device),
        torch.tensor(np.array(ys), dtype=torch.float32, device=device),
    )


def train_model(buffer: ReplayBuffer):
    model.train()
    optimizer.zero_grad()

    batch_idxs = buffer.sample(MINI_BATCH_SIZE)

    modality_frames, modality_types, ys = sample(buffer, batch_idxs)
    prediction = model(modality_frames, modality_types)

    loss = F.huber_loss(prediction, ys)
    loss.backward()
    optimizer.step()

    buffer[batch_idxs] = loss.item()

    model.eval()
    return loss.item()


def train_normaliser(buffer: ReplayBuffer):
    pass
    # bounds.fit(buffer).save("bounds.json")


if __name__ == "__main__":
    # todo write a new way to create normalisation bounds. currently we have no way to fit the bounds
    bounds = NormalisationBounds().load("bounds.json")

    model = PositionalModelling(10, EMBEDDING_SIZE, n_layers=N_LAYERS).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    env = MultiEnvironment(
        subset="training",
        envs=N_ENVIRONMENTS,
        seq_len=SEQUENCE_LENGTH,
        random_subsample=True
    )

    positions = env.reset()
    # Start sample from the environment
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    episode_step_error = []
    episode_losses = []

    mlflow.set_experiment("positioning")

    with mlflow.start_run(run_name="Layers {}x{}, MB {} RBS {}".format(N_LAYERS, EMBEDDING_SIZE, MINI_BATCH_SIZE, REPLAY_BUFFER_SIZE)):
        mlflow.log_params({
            "sequence_length": SEQUENCE_LENGTH,
            "mini_batch_size": MINI_BATCH_SIZE,
            "max_steps": MAX_STEPS,
            "learning_rate": LEARNING_RATE,
            "replay_buffer_size": REPLAY_BUFFER_SIZE,
            "n_environments": N_ENVIRONMENTS,
            "gaussian_noise_factor": GAUSSIAN_NOISE_FACTOR,
            "embedding_size": EMBEDDING_SIZE,
            "n_layers": N_LAYERS,
        })

        model.eval()
        for step in range(0, MAX_STEPS):
            env_id = step % len(env)
            percentage_complete = step / MAX_STEPS
            percentage_remaining = 1 - percentage_complete

            # Scale the learning rate with the percentage_complete
            scale = 1 - math.pow(step / MAX_STEPS, 0.1)
            for g in optimizer.param_groups:
                g['lr'] = LEARNING_RATE * scale

            state, position_change, termination = env.step(env_id, positions[env_id])

            # Process and create the model input to what the target change should be then predict the position for it
            modality_data, modality_types = process_state(
                state,
                seq_length=SEQUENCE_LENGTH,
                normalisation_bounds=bounds
            )
            with torch.no_grad():
                predicted_position_change = model(
                    torch.tensor(modality_data).to(device=device, dtype=torch.float32),
                    torch.tensor(modality_types).to(device=device, dtype=torch.int64),
                )
                predicted_position_change = predicted_position_change.detach().cpu().numpy()[0] * MAX_SNAP_GPS_DISTANCE

            # To prevent errors exploding during instances where the drift increases, if the error is too high,
            # then we apply the position fix and effectively reset the trajectory
            # Doing so means we can keep training during this run
            step_error = np.linalg.norm(position_change - predicted_position_change)
            episode_step_error.append(step_error)
            buffer.append(
                TrainingSample(
                    model_input=(modality_data, modality_types),
                    target=position_change / MAX_SNAP_GPS_DISTANCE,
                ),
                F.huber_loss(
                    torch.tensor(predicted_position_change / MAX_SNAP_GPS_DISTANCE),
                    torch.tensor(position_change / MAX_SNAP_GPS_DISTANCE)
                ).item()
            )

            # Update the noise with some noise
            position_noise_factor = percentage_remaining * GAUSSIAN_NOISE_FACTOR
            position_noise = np.random.uniform(-1, 1, positions.shape) * position_noise_factor
            positions[env_id] += (
                  position_change
                  if step_error > MAX_SNAP_GPS_DISTANCE else
                  predicted_position_change
            ) + position_noise[env_id]

            if len(buffer) == INITIAL_TRAIN_SIZE:
                train_normaliser(buffer)

            mini_batch_loss = None
            if len(buffer) >= MINI_BATCH_SIZE:
                mini_batch_loss = train_model(buffer)
                episode_losses.append(mini_batch_loss)

            # Reset env if terminated
            if termination:
                positions = env.reset(env_id)

            # Log metrics to mlflow
            mlflow.log_metric("step_error", step_error, step=step)
            mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=step)
            if mini_batch_loss is not None:
                mlflow.log_metric("loss", mini_batch_loss, step=step)

            # Print a status update every x steps
            if (step + 1) % LOG_EVERY_N_STEPS == 0:
                mean_step_err = np.mean(episode_step_error)
                mean_loss = np.mean(episode_losses) if episode_losses else 0.0
                print("\r{} Mean Step {:.5f} Loss {:.5f}".format(
                    step + 1,
                    mean_step_err,
                    mean_loss
                ))
                mlflow.log_metric("mean_step_error_window", mean_step_err, step=step)
                mlflow.log_metric("mean_loss_window", mean_loss, step=step)
                episode_step_error = []
                episode_losses = []
                if (step + 1) % SAVE_EVERY_N_STEPS == 0:
                    model_path = f"model-{step + 1}.pt"
                    torch.save(model.state_dict(), model_path)
                    # mlflow.log_artifact(model_path)

            elif step % 10 == 0:
                print("\r{} Mean Step {:.5f} Loss {:.5f} {:.5f}".format(
                    step,
                    np.mean(episode_step_error),
                    np.mean(episode_losses) if episode_losses else 0.0,
                    buffer.mean_loss()
                ), end="")
