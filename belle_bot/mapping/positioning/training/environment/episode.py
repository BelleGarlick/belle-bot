import json
import os
import random

import numpy as np

from belle_bot.mapping.positioning.training.models import GpsPoint, ImuData

REPLAY_FILE_PATH = "/Users/belle/Developer/belle-bot/replays"

# todo create multi-agent environment which is just one container with N number of child environments within


def open_replay_file(replay_file):
    with open(os.path.join(REPLAY_FILE_PATH, replay_file), "r") as f:
        return f.readlines()


def _parse_events(replay_id: str):
    lines = open_replay_file(replay_id)

    events = []
    for i, line in enumerate(lines):
        split_tokens = line.split(",")
        stream = split_tokens[0]
        timestamp = float(split_tokens[1])
        data = json.loads(",".join(split_tokens[2:]))

        if stream == "sensors/gps":
            if data['has_fix'] == "True":
                events.append(GpsPoint.from_data(timestamp, data))
                events[-1].name = str(i)

        # if stream == "sensors/camera":
        #     events.append(
        #         CameraData.from_data(timestamp, data)
        #     )

        if stream == "sensors/imu":
            events.append(
                ImuData.from_data(timestamp, data)
            )
            events[-1].name = str(i)

    return events


def _subsample_events(events):
    half_length = int(len(events) / 2)
    start_index = random.randint(0, half_length)

    return events[start_index:half_length + start_index]


def _create_event_pairs(events):
    """
    This function will iterate through the events and find the previous gps
    and next gps point which will be used to ground the model against

    :param events: The events list
    :return: The pairs of events
    """
    event_pairs = [[e] for e in events]

    gps_queue = [None, None]
    for i in range(len(event_pairs)):
        item = event_pairs[i][0]
        if isinstance(item, GpsPoint):
            gps_queue.append(event_pairs[i][0])
            gps_queue = gps_queue[-2:]

        event_pairs[i] = event_pairs[i] + [gps_queue[0], gps_queue[1]]

    gps_queue = [None, None]
    for i in range(len(event_pairs) - 1, -1, -1):
        item = event_pairs[i][0]
        if isinstance(item, GpsPoint):
            gps_queue.append(event_pairs[i][0])
            gps_queue = gps_queue[-2:]

        event_pairs[i] = event_pairs[i] + [gps_queue[1], gps_queue[0]]

    return [tuple(x) for x in event_pairs if all(x[1:])]


def _get_initial_gps_pos(events):
    gps_count = 0
    for i in range(len(events)):
        item = events[i]
        if isinstance(item, GpsPoint):
            gps_count += 1

        if gps_count == 2:
            return item, i

    return None



def calculate_catmull_rom_segment(p0, p1, p2, p3, t) -> np.ndarray:
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


class Episode:
    def __init__(self, replay_path, random_subsample=False):
        self.replay_path = replay_path
        self.random_subsample = random_subsample

        self.events: list[tuple[ImuData | GpsPoint, GpsPoint, GpsPoint, GpsPoint, GpsPoint]] = []
        self.current_step_idx = None

    def reset(self) -> np.ndarray:
        """

        :return: Global position
        """
        # todo not sure how to handle this for future, but for now we will just return the first gps position in the list to get the position. this wont work for cold starts on the model itself
        events = _parse_events(self.replay_path)
        if self.random_subsample:
            events = _subsample_events(events)
        self.events = _create_event_pairs(events)

        self.current_step_idx = 0

        return self.calculate_position(self.current_step_idx)

    def calculate_position(self, idx) -> np.ndarray:
        item, gps0, gps1, gps2, gps3 = self.events[idx]

        # If gps then just return the global point
        if isinstance(item, GpsPoint):
            return item.numpy()

        if not (gps0.timestamp < gps1.timestamp <= item.timestamp <= gps2.timestamp < gps3.timestamp):
            breakpoint()

        # If not gps then create the position that it should be by interpolating between
        t = (item.timestamp - gps1.timestamp) / (gps2.timestamp - gps1.timestamp)
        if t < 0 or t > 1:
            breakpoint()

        interpolated_xy = calculate_catmull_rom_segment(
            np.array([gps0.x, gps0.y]),
            np.array([gps1.x, gps1.y]),
            np.array([gps2.x, gps2.y]),
            np.array([gps3.x, gps3.y]),
            t=t
        )
        interpolated_t = (gps2.altitude - gps1.altitude) * t + gps1.altitude

        return np.array([
            interpolated_xy[0],
            interpolated_xy[1],
            interpolated_t,
        ])


    def step(self):
        """
        Move to the next step

        :return: Current step
        """
        self.current_step_idx += 1

        return (
            self.events[self.current_step_idx][0],
            self.calculate_position(self.current_step_idx),
            self.current_step_idx == len(self.events) - 1  # check if terminated
        )
