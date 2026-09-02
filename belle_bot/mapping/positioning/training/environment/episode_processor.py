import copy
import math
import random
from typing import Literal

import numpy as np

from belle_bot.houston.client.py import replays
from belle_bot.mapping.positioning.training.environment.env import Frame
from belle_bot.mapping.positioning.training.environment.episode import Episode
from belle_bot.mapping.positioning.training.models import GpsPoint


def load_replay_ids(subset: Literal['training', 'testing'] | None) -> list[str]:
    filter = ["dataset/mapping/position"]
    if subset:
        filter += [subset]

    replay_ids = replays.query_replays(
        page=0,
        tags=filter
    )['replays']

    return sorted([x["replay_id"] for x in replay_ids])



def process_episode(episode: Episode):
    """Takes an episode and converts it to the list of steps that define the current step

    :param episode: The episode to process
    :return:
    """
    data = []

    # due to how the step cycle is run, we miss the first state transition

    while True:
        frame, old_position, new_position, time_delta, terminated = episode.step()

        pre_position_change = np.zeros((3,))
        if data:
            pre_position_change = data[-1]["delta"]

        frame = copy.deepcopy(frame)
        if isinstance(frame, GpsPoint):
            frame.x -= old_position[0]
            frame.y -= old_position[1]
            frame.altitude -= old_position[2]

        data.append(Frame(
            frame=frame,
            prev_position=old_position,
            true_position=new_position,
            position_change=new_position - old_position,
            prev_position_change=np.zeros(3),
            time_delta=time_delta,
        ))

        if terminated:
            break

    return data


# todo
#  eventually turn into web dataset
#  eventually make it so we can have randomness during this so the model can deal with imperfect data

def load_episodes(subset: Literal['training', 'testing'], limit=None, augment_rotation: int | None = None):
    replay_ids = load_replay_ids(subset)

    episodes = []
    for replay_id in replay_ids:
        if augment_rotation is not None:
            for _ in range(augment_rotation):
                episodes.append(Episode(replay_id, random_subsample=False, rotation_angle=random.random() * math.tau))
        else:
            episodes.append(Episode(replay_id, random_subsample=False))

        if limit is not None and len(episodes) >= limit:
            break

    return episodes


def process_episodes(episodes: list[Episode], seq_len: int = 100):
    windows = []
    for episode in episodes:
        episode_frames = process_episode(episode)
        for i in range(1, len(episode_frames)):
            windows.append(episode_frames[max(0, i-seq_len):i])

    return windows


def load_dataset(subset: Literal['training', 'testing'], seq_len: int = 100, limit: int | None = None, augment_rotation: int | None = None):
    episodes = load_episodes(subset, limit, augment_rotation=augment_rotation)
    return process_episodes(episodes, seq_len)


if __name__ == "__main__":
    load_dataset('training')
    load_dataset('testing')
