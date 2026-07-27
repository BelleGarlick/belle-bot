import json
import os

import numpy as np

from belle_bot.mapping.positioning.training.models import GpsPoint, GPSReplay, ImuData, CameraData

# predict position and confidence
# need to predict with missing data throughout the input


# TODO
#  Have a way to interpolate between points
#  Possible have a way to have just a datum which gets put into the model without a specific embedding. it just has an embedding
#  include camera
#  have a way to deteriorate the input data so it's not as perfect
#  should create a webdataset and upload it to houston
#  change it so we have a way to predict position changes even if missing gps position



REPLAY_FILE_PATH = "/Users/belle/Developer/belle-bot/replays"
replay_files = os.listdir(REPLAY_FILE_PATH)


VALID_FILES = {
    "14240eda-3dfc-48e0-921b-c22397ca0981.txt",
    "5ba31b5c-b457-4c80-9dc4-2f42d9f61a9f.txt",
    # "87d554ee-3d54-4440-85cb-756f0c234067.txt"
}


def calculate_catmull_rom_segment(p0, p1, p2, p3, t):
    """Calculates points on a single Catmull-Rom segment between p1 and p2."""
    # T can be a fixed point or an array
    # Catmull-Rom characteristic matrix coefficients
    # Formula: 0.5 * ((2*p1) + (-p0 + p2)*t + (2*p0 - 5*p1 + 4*p2 - p3)*t^2 + (-p0 + 3*p1 - 3*p2 + p3)*t^3)
    point = 0.5 * (
            (2 * p1) +
            (-p0 + p2) * t +
            (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t ** 2) +
            (-p0 + 3 * p1 - 3 * p2 + p3) * (t ** 3)
    )
    return point


def open_replay_file(replay_file):
    with open(os.path.join(REPLAY_FILE_PATH, replay_file), "r") as f:
        return f.readlines()


def parse_data(lines):
    events = []

    for line in lines:
        split_tokens = line.split(",")
        stream = split_tokens[0]
        timestamp = float(split_tokens[1])
        data = json.loads(",".join(split_tokens[2:]))

        if stream == "sensors/gps":
            if data['has_fix']:
                events.append(GpsPoint.from_data(timestamp, data))

        # if stream == "sensors/camera":
        #     events.append(
        #         CameraData.from_data(timestamp, data)
        #     )

        if stream == "sensors/imu":
            events.append(
                ImuData.from_data(timestamp, data)
            )

    return GPSReplay(events=events)


def chunk_replay_file(replay_file: GPSReplay, max_events=100):
    gps_points = [x for x in replay_file.events if isinstance(x, GpsPoint)]

    # todo drop replay events here so that we can train on the model with missing data
    frames = []
    for gps_idx in range(2, len(gps_points) - 2):
        interp = np.random.random()
        target_point = calculate_catmull_rom_segment(
            gps_points[gps_idx - 1].numpy(),
            gps_points[gps_idx].numpy(),
            gps_points[gps_idx + 1].numpy(),
            gps_points[gps_idx + 2].numpy(),
            t=interp
        )
        timestamp = (gps_points[gps_idx + 1].timestamp - gps_points[gps_idx].timestamp) * interp\
                    + gps_points[gps_idx].timestamp

        # Create the target point
        target_point = GpsPoint(
            timestamp=timestamp,
            x=target_point[0],
            y=target_point[1],
            altitude=target_point[2],
        )

        # Get the 100 events before the current point
        events = [x for x in replay_file.events if x.timestamp <= target_point.timestamp]
        events = events[-max_events:]

        frames.append(
            GPSReplay(
                events=events,
                target=target_point
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
