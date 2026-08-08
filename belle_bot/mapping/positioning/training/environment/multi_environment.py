import numpy as np
from typing_extensions import Literal

import houston_api_client
from belle_bot.mapping.positioning.training.environment import Environment
from houston_api_client.api.replays import list_replays


houston_client = houston_api_client.Client(base_url="http://localhost:8081/")

def load_replay_ids(subset: Literal['training', 'testing']) -> list[int]:
    # Load by tag
    response = list_replays.sync(client=houston_client, page=0, tags=[subset, "dataset/mapping/position"])
    return [x.replay_id for x in response.replays]



class MultiEnvironment:

    def __init__(self, subset: Literal['training', 'testing'], seq_len, envs=8, random_subsample=False):
        self.replay_ids = load_replay_ids(subset=subset)

        self._new_replay_idx = -1
        self.env_count = envs
        self.seq_len = seq_len
        self.random_subsample = random_subsample

        self.envs: list[Environment] = []
        self.__positions: list[np.ndarray] = []

    def __len__(self):
        return len(self.envs)

    def __get_new_replay_idx(self):
        self._new_replay_idx += 1
        return self._new_replay_idx % len(self.replay_ids)

    def reset(self, idx=None) -> np.ndarray:
        if idx is None:
            if self.envs:
                self.envs.clear()
                self.__positions.clear()

            # If no idx, then reset all environments
            for idx in range(self.env_count):
                replay_id = self.replay_ids[self.__get_new_replay_idx()]
                env = Environment(replay_id, self.seq_len, self.random_subsample)
                self.envs.append(env)
                self.__positions.append(env.current_episode.current_position())

            return np.array(self.__positions)

        # Create a new episode and reset it to get the initial position
        replay_id = self.replay_ids[self.__get_new_replay_idx()]
        self.envs[idx] = Environment(replay_id, self.seq_len, self.random_subsample)
        self.__positions[idx] = self.envs[idx].current_episode.current_position()
        return np.array(self.__positions)

    def step(self, idx, position: np.ndarray):
        return self.envs[idx].step(position)
