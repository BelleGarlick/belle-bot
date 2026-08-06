from collections import deque
from dataclasses import dataclass

import numpy as np

import torch
import torch.nn.functional as F

from belle_bot.mapping.positioning.training.environment.multi_environment import MultiEnvironment
from belle_bot.mapping.positioning.training.ml_model import PositionalModelling
from belle_bot.mapping.positioning.training.models import GpsPoint, ModalityEnum, ImuData
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds

# todo
#  estimate variance when training
#  drop sections randomly
#  create normalisation from some initial steps
#  mlflow
#  roate the scene
#  add camera
#  bspline gps
#  cli args to trigger training runs


SEQUENCE_LENGTH = 100
MINI_BATCH_SIZE = 256
LOG_EVERY_N_STEPS = 5000
SAVE_EVERY_N_STEPS = 10_000
MAX_SNAP_GPS_DISTANCE = 10
INITIAL_TRAIN_SIZE = 500  # used to accumulate data for normalisation
REPLAY_BUFFER_SIZE = MINI_BATCH_SIZE
N_ENVIRONMENTS = 20
GAUSSIAN_NOISE_FACTOR = 0.1
MAX_STEPS = 1000000
LEARNING_RATE = 1e-3


@dataclass
class TrainingSample:
    model_input: tuple[np.ndarray, np.ndarray]
    target: np.ndarray


class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.buffer = deque[TrainingSample](maxlen=maxlen)
        self.buffer_errors = deque[float](maxlen=maxlen)

    def sample(self, n):
        errors = softmax(np.array(self.buffer_errors))
        return np.random.choice(len(self.buffer), n, p=errors)

    def __getitem__(self, idxs):
        return self.buffer[idxs]

    def __setitem__(self, idxs, val):
        for idx in idxs:
            # anneal the value towards the true value
            self.buffer_errors[idx] = self.buffer_errors[idx] * 0.5 + val * 0.5

    def __len__(self):
        return len(self.buffer)

    def append(self, sample, error):
        self.buffer.append(sample)
        self.buffer_errors.append(error)

    def mean_loss(self):
        return np.mean(np.array(self.buffer_errors))

    def clear(self):
        self.buffer.clear()
        self.buffer_errors.clear()


# instead, sample more items, but only train on the items where the error is larger. so it becomes a sort of heirstic search. doing so means we're not wasting cycles train pointeless data.

device = torch.device('mps')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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

    model = PositionalModelling(10, 320).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    env = MultiEnvironment(envs=N_ENVIRONMENTS, seq_len=SEQUENCE_LENGTH, random_subsample=True)

    positions = env.reset()
    # Start sample from the environment
    buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    episode_step_error = []
    episode_losses = []

    model.eval()
    for step in range(0, MAX_STEPS):
        env_id = step % len(env)
        percentage_complete = step / MAX_STEPS
        percentage_remaining = 1 - percentage_complete

        # Scale the learning rate with the percentage_complete
        for g in optimizer.param_groups:
            g['lr'] = LEARNING_RATE * percentage_remaining

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

        if len(buffer) >= MINI_BATCH_SIZE:
            mini_batch_loss = train_model(buffer)
            # buffer.clear()
            episode_losses.append(mini_batch_loss)

        # Reset any envs that terminated
        if termination:
            positions[env_id] = env.reset(env_id)

        # Print a status update every x steps
        if (step + 1) % LOG_EVERY_N_STEPS == 0:
            print("\r{} Mean Step {:.5f} Loss {:.5f}".format(
                step + 1,
                np.mean(episode_step_error),
                np.mean(episode_losses)
            ))
            episode_step_error = []
            episode_losses = []
            if (step + 1) % SAVE_EVERY_N_STEPS == 0:
                torch.save(model.state_dict(), f"model-{step + 1}.pt")
        elif step % 10 == 0:
            print("\r{} Mean Step {:.5f} Loss {:.5f} {:.5f}".format(
                step,
                np.mean(episode_step_error),
                np.mean(episode_losses),
                buffer.mean_loss()
            ), end="")
