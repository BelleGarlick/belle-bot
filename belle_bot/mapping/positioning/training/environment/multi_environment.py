import numpy as np
from attr import dataclass
from typing_extensions import Literal

from belle_bot.mapping.positioning.training.environment import Environment, Episode
from belle_bot.mapping.positioning.training.environment.env import Frame
from belle_bot.mapping.positioning.training.models import GpsPoint, ImuData
from belle_bot.houston.client.py import replays


def load_replay_ids(subset: Literal['training', 'testing'] | None) -> list[str]:
    filter = ["dataset/mapping/position"]
    if subset:
        filter += [subset]

    replay_ids = replays.query_replays(
        page=0,
        tags=filter
    )['replays']

    return sorted([x["replay_id"] for x in replay_ids])


@dataclass
class ResetData:
    initial_states: list[Frame]
    replay_ids: list[str]


class MultiEnvironment:

    def __init__(self, subset: Literal['training', 'testing'] | None, seq_len, envs=8, random_subsample=False):
        self.replay_ids: list[str] = load_replay_ids(subset=subset)

        self._new_replay_idx = -1
        self.env_count = envs
        self.seq_len = seq_len
        self.random_subsample = random_subsample

        self.environments: list[Environment] = []

    def __len__(self):
        return len(self.environments)

    def __get_new_replay_id(self):
        self._new_replay_idx += 1
        replay_id = self._new_replay_idx % len(self.replay_ids)
        return self.replay_ids[replay_id]

    def reset(self, idx: int | None=None) -> ResetData:
        if idx is None:
            if self.environments:
                self.environments.clear()

            # If no idx, then reset all environments
            replay_ids = []
            for idx in range(self.env_count):
                replay_id = self.__get_new_replay_id()
                replay_ids.append(replay_id)
                self.environments.append(
                    Environment(
                        Episode(replay_id, self.random_subsample),
                        self.seq_len
                    )
                )

            return ResetData(replay_ids=replay_ids, initial_states=[x.reset() for x in self.environments])

        # Create a new episode and reset it to get the initial position
        replay_id = self.__get_new_replay_id()
        self.environments[idx] = Environment(
            Episode(replay_id, self.random_subsample),
            self.seq_len
        )

        return ResetData(
            replay_ids=[replay_id],
            initial_states=[self.environments[idx].reset()]
        )

    def step(
            self,
            idx: int,
            action,
            max_error: float | None = None
    ) -> tuple[ImuData | GpsPoint, np.ndarray, bool]:
        return self.environments[idx].step(action, max_error)
