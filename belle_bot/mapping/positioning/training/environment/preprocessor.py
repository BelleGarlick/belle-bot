import numpy as np
import torch

from belle_bot.mapping.positioning.training.environment.env import Frame
from belle_bot.mapping.positioning.training.models import ImuData, ModalityEnum, GpsPoint
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds


def encode_frame(item, normalisation_bounds: NormalisationBounds) -> tuple[np.ndarray, ModalityEnum]:
    if isinstance(item.frame, ImuData):
        # Update the modality data
        # todo change the angle to cos / sin so that it's measured a lil better at the point from -180 to 180
        return (
            np.hstack((
                item.prev_position_change / 10,
                item.frame.acc / normalisation_bounds['imu.acc'],
                item.frame.gyro / normalisation_bounds['imu.gyro'],
                item.frame.angle / normalisation_bounds['imu.angle'],
                [item.time_delta]
            )),
            ModalityEnum.IMU,
        )

    elif isinstance(item.frame, GpsPoint):
        # Update the modality data
        return (
            np.array((item.prev_position_change / 10).tolist() + [
                min(10, item.frame.x / normalisation_bounds["gps.x"]),
                min(10, item.frame.y / normalisation_bounds["gps.y"]),
                min(10, item.frame.altitude / normalisation_bounds["gps.alt"]),
            ] + [0] * 6 + [item.time_delta]), # padded the item so everything is same size
            ModalityEnum.GPS
        )

    else:
        # split up camera into multiple tokens
        raise NotImplementedError()


# todo also make make it so that eventually we can make something more stochastic where frames are dropped / not perfectly discrete.
# todo update the testing

# realistically, i think the best thing to do is to jus use frames which embed the diff item
def process_state(frames: list[Frame], seq_length, normalisation_bounds: NormalisationBounds):
    modality_types = [ModalityEnum.PAD] * seq_length
    modality_data = [np.array([0] * 13) for _ in range(seq_length)]

    for item in frames:
        data, type = encode_frame(item, normalisation_bounds)
        modality_data.append(data)
        modality_types.append(type)

    modality_types = np.array([modality_types[-seq_length:]], dtype=np.int64)
    modality_data = np.array([modality_data[-seq_length:]], dtype=np.float32)

    # if modality_data.max() > 30 or modality_data.min() < -30:
    #     breakpoint()

    return (
        modality_data,
        modality_types
    )


def process_frames(data, normalisation_bounds: NormalisationBounds, seq_length: int, device: torch.device):
    x_f, x_m, y = [], [], []

    for item in data:
        x = process_state(item, seq_length, normalisation_bounds)
        x_f.append(x[0])
        x_m.append(x[1])
        if 'delta' in item[-1]:
            y.append(item[-1]['delta'])

    return (
        torch.tensor(np.vstack(x_f), dtype=torch.float32, device=device),
        torch.tensor(np.vstack(x_m), dtype=torch.int64, device=device),
        torch.tensor(np.array(y), dtype=torch.float32, device=device),
    )

