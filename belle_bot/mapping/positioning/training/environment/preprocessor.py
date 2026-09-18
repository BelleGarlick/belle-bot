import numpy as np
import torch

from belle_bot.mapping.positioning.config.positioning_config import PositioningConfig
from belle_bot.mapping.positioning.training.environment.env import Frame
from belle_bot.mapping.positioning.training.models import ImuData, ModalityEnum, GpsPoint, CameraData
from belle_bot.mapping.positioning.training.normalisation import NormalisationBounds


def encode_frame(config: PositioningConfig, item, normalisation_bounds: NormalisationBounds) -> tuple[np.ndarray, list[ModalityEnum]]:
    frame_size = 13
    if config.model.include_camera:
        chunk_size = config.model.camera_chunk_size
        frame_size = (chunk_size * chunk_size) + 5

    if isinstance(item.frame, ImuData):
        # todo change the angle to cos / sin so that it's measured a lil better at the point from -180 to 180

        # Update the modality data
        frame = np.zeros((1, frame_size))

        frame[0, 0] = item.time_delta
        frame[0, 1:4] = item.prev_position_change / 10
        frame[0, 4:7] = item.frame.acc / normalisation_bounds['imu.acc']
        frame[0, 7:10] = item.frame.gyro / normalisation_bounds['imu.gyro']
        frame[0, 10:13] = item.frame.angle / normalisation_bounds['imu.angle']

        return (
            frame, # padded the item so everything is same size
            [ModalityEnum.IMU]
        )

    elif isinstance(item.frame, GpsPoint):
        # Update the modality data
        frame = np.zeros((1, frame_size))

        frame[0, 0] = item.time_delta
        frame[0, 1:4] = item.prev_position_change / 10
        frame[0, 4] = min(10, item.frame.x / normalisation_bounds["gps.x"])
        frame[0, 5] = min(10, item.frame.y / normalisation_bounds["gps.y"])
        frame[0, 6] = min(10, item.frame.altitude / normalisation_bounds["gps.alt"])

        return (
            frame, # padded the item so everything is same size
            [ModalityEnum.GPS]
        )

    elif isinstance(item.frame, CameraData):
        # Take the frame and split into a series of subframes
        im_height, im_width = item.frame.frame.shape
        chunk_size = config.model.camera_chunk_size
        chunks_count_x = im_height // chunk_size
        chunks_count_y = im_width // chunk_size
        chunk_count = chunks_count_x * chunks_count_y

        chunks = np.zeros((chunk_count, frame_size))

        chunks[:, 0] = item.time_delta
        chunks[:, 1:4] = item.prev_position_change / 10

        chunk_i = 0
        for chunk_y in range(0, im_height - chunk_size, chunk_size):
            for chunk_x in range(0, im_width - chunk_size, chunk_size):
                chunk = item.frame.frame[chunk_y:chunk_y + chunk_size,chunk_x:chunk_x + chunk_size]
                chunk = chunk.flatten()

                chunks[chunk_i, 4] = chunk_i
                chunks[chunk_i, 5:] = chunk

                chunk_i += 1

        chunks[:, 3] / chunk_i

        return chunks, [ModalityEnum.CAMERA] * chunk_count

    else:
        # split up camera into multiple tokens
        raise NotImplementedError()


# todo also make make it so that eventually we can make something more stochastic where frames are dropped / not perfectly discrete.
# todo update the testing

# realistically, i think the best thing to do is to jus use frames which embed the diff item
def process_state(config: PositioningConfig, frames: list[Frame], normalisation_bounds: NormalisationBounds):
    seq_length = config.model.sequence_length

    modality_types = []
    modality_data = []

    for item in frames:
        data, type = encode_frame(config, item, normalisation_bounds)
        modality_data.append(data)
        modality_types += type

    modality_types = modality_types[-seq_length:]
    while len(modality_types) < seq_length:
        modality_types.insert(0, ModalityEnum.PAD)

    modality_data = np.concatenate(modality_data, axis=0)[-seq_length:]
    full_modality_data = np.zeros((seq_length, modality_data.shape[-1]))
    full_modality_data[-modality_data.shape[0]:] = modality_data

    modality_types = np.array([modality_types], dtype=np.int64)
    modality_data = np.expand_dims(full_modality_data, axis=0)

    return (
        modality_data,
        modality_types
    )


def process_frames(config, data, normalisation_bounds: NormalisationBounds, device: torch.device):
    x_f, x_m, y = [], [], []

    for item in data:
        x = process_state(config, item, normalisation_bounds)
        x_f.append(x[0])
        x_m.append(x[1])
        if 'delta' in item[-1]:
            y.append(item[-1]['delta'])

    return (
        torch.tensor(np.vstack(x_f), dtype=torch.float32, device=device),
        torch.tensor(np.vstack(x_m), dtype=torch.int64, device=device),
        torch.tensor(np.array(y), dtype=torch.float32, device=device),
    )

