import base64
import json
import os
from dataclasses import dataclass

import numpy as np

from belle_bot.mapping.positioning.training.models import GpsPoint


# predict position and confidence
# need to predict with missing data throughout the input


# TODO
#  Have a way to interpolate between points
#  Possible have a way to have just a datum which gets put into the model without a specific embedding. it just has an embedding
#  include camera
#  have a way to deteriorate the input data so it's not as perfect
#  should create a webdataset and upload it to houston


@dataclass
class ImuData:
    timestamp: int
    gyro: np.ndarray
    acc: np.ndarray
    angle: np.ndarray

    @staticmethod
    def from_data(timestamp, data):
        parse_datum = lambda key: np.frombuffer(base64.b64decode(data[key]), dtype=np.float32)

        return ImuData(
            timestamp=timestamp,
            acc=parse_datum("acc"),
            gyro=parse_datum("gyro"),
            angle=parse_datum("angle"),
        )


@dataclass
class GPSReplay:
    gps: list[GpsPoint]
    imu: list[ImuData]
    # camera: list[CameraFrame]


REPLAY_FILE_PATH = "/Users/belle/Developer/belle-bot/replays"
replay_files = os.listdir(REPLAY_FILE_PATH)


VALID_FILES = {
    "14240eda-3dfc-48e0-921b-c22397ca0981.txt",
    "5ba31b5c-b457-4c80-9dc4-2f42d9f61a9f.txt",
    # "87d554ee-3d54-4440-85cb-756f0c234067.txt"
}


def open_replay_file(replay_file):
    with open(os.path.join(REPLAY_FILE_PATH, replay_file), "r") as f:
        return f.readlines()


def parse_data(lines):
    gps_coords = []
    imu_data = []

    for line in lines:
        split_tokens = line.split(",")
        stream = split_tokens[0]
        timestamp = float(split_tokens[1])
        data = json.loads(",".join(split_tokens[2:]))

        if stream == "sensors/gps":
            if data['has_fix']:
                gps_coords.append(GpsPoint.from_data(timestamp, data))

        if stream == "sensors/camera":
            pass

        if stream == "sensors/imu":
            imu_data.append(
                ImuData.from_data(timestamp, data)
            )

    return GPSReplay(
        gps=gps_coords,
        imu=imu_data,
    )


def chunk_replay_file(replay_file):
    # todo eventually make a better way to interpolate between
    #  to begin with we will just take all the gps points and take the previous 20s
    # todo eventually just put everything into a trainable embedding so that we can just put data in as it gets it rather than having a script format
    # ideally, eventually we just take all data, embed whatever we have and put it in the model to output the position with it's confidence
    frames = []
    for point in replay_file.gps[20:]:
        gps_points = [
            x for x in replay_file.gps
            if point.timestamp > x.timestamp > point.timestamp - 20
        ]
        imu_points = [
            x for x in replay_file.imu
            if point.timestamp > x.timestamp > point.timestamp - 20
        ]

        frames.append(
            GPSReplay(
                gps=gps_points,
                imu=imu_points,
            )
        )

    return frames


# TODO create dataset eventually that is stored in the houston thing
def load_dataset():
    all_chunks = []
    for replay_file in replay_files:
        if replay_file not in VALID_FILES: continue
        if replay_file[0] == ".": continue
        lines = open_replay_file(replay_file)

        replay_file = parse_data(lines)

        # TODO add smoothing to the line so we can interpolate at any point

        all_chunks += chunk_replay_file(replay_file)

    idxs = np.arange(len(all_chunks))
    np.random.shuffle(idxs)
    val_length = int(len(all_chunks) * 0.2)

    train_set = [all_chunks[idx] for idx in idxs[val_length:]]
    val_set = [all_chunks[idx] for idx in idxs[:val_length]]

    return train_set, val_set
