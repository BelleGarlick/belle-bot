import numpy as np
from collections import deque
from dataclasses import dataclass


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()



@dataclass
class TrainingSample:
    model_input: tuple[np.ndarray, np.ndarray]
    target: np.ndarray



class ReplayBuffer:
    def __init__(self, maxlen: int):
        self.buffer = deque[TrainingSample](maxlen=maxlen)
        self.buffer_errors = deque[float](maxlen=maxlen)

    def sample(self, n, seed=None):
        # Higher temperature (e.g., 0.5 - 1.0) prevents extreme probability collapse
        # while still prioritizing higher-error experiences
        temperature = 0.5

        # Scale errors and apply softmax safely
        scaled_errors = np.array(self.buffer_errors) / temperature

        # Subtract max for numerical stability before exponentiation
        exp_errors = np.exp(scaled_errors - np.max(scaled_errors))
        errors = exp_errors / np.sum(exp_errors)

        if seed is not None:
            rng = np.random.default_rng(seed)
            return rng.choice(len(self.buffer), size=n, replace=False, p=errors)

        # replace=False guarantees unique indices
        return np.random.choice(len(self.buffer), size=n, replace=False, p=errors)

    def __getitem__(self, idxs):
        if isinstance(idxs, (list, np.ndarray)):
            return [self.buffer[i] for i in idxs]
        return self.buffer[idxs]

    def __setitem__(self, idxs, vals):
        for idx, val in zip(idxs, vals):
            # anneal the value towards the true value
            self.buffer_errors[idx] = self.buffer_errors[idx] * 0.5 + val * 0.5

    def __len__(self):
        return len(self.buffer)

    def append(self, sample, error=1):
        self.buffer.append(sample)
        self.buffer_errors.append(error)

    def mean_loss(self):
        return np.mean(np.array(self.buffer_errors))

    def clear(self):
        self.buffer.clear()
        self.buffer_errors.clear()
