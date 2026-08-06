from collections import deque
from dataclasses import dataclass

from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.ModuleState import np


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

    def sample(self, n):
        errors = softmax(np.array(self.buffer_errors))
        return np.random.choice(len(self.buffer), n, p=errors)

    def __getitem__(self, idxs):
        return self.buffer[idxs]

    def __setitem__(self, idxs, val):
        for idx in idxs:
            # anneal the value towards the true value
            self.buffer_errors[idx] = self.buffer_errors[idx] * 0.5 + val * 0.5

    def __len__(self):
        return len(self.buffer)

    def append(self, sample, error):
        self.buffer.append(sample)
        self.buffer_errors.append(error)

    def mean_loss(self):
        return np.mean(np.array(self.buffer_errors))

    def clear(self):
        self.buffer.clear()
        self.buffer_errors.clear()
