import numpy as np

from belle_bot.mapping.positioning.training.environment import Environment


class MultiEnvironment:

    def __init__(self, seq_len, envs=8, random_subsample=False):
        self.envs = [
            Environment(seq_len=seq_len, random_subsample=random_subsample)
            for _ in range(envs)
        ]

    def __len__(self):
        return len(self.envs)

    def reset(self, idx=None) -> np.ndarray:
        if idx is None:
            return np.array([env.reset() for env in self.envs])

        return self.envs[idx].reset()

    def step(self, idx, position: np.ndarray):
        return self.envs[idx].step(position)
