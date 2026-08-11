import copy
import dataclasses
from typing import Tuple

import numpy as np

from belle_bot.mapping.positioning.training.environment.episode import Episode
from belle_bot.mapping.positioning.training.models import GpsPoint, ImuData


@dataclasses.dataclass
class Frame:
    frame: GpsPoint | ImuData
    time_delta: float
    position_change: np.ndarray # position change to get to this state
    current_position: np.ndarray


class Environment:

    def __init__(self, episode: Episode, seq_len: int):
        self.episode = episode
        self.seq_len = seq_len

        self.position: np.ndarray = np.zeros(3)

        self.visited_frames: list[Frame] = []

        self.last_timestep = 0

    def reset(self):
        frame, position, _ = self.episode.step()
        self.position = self.episode.current_position()
        self.last_timestep = self.episode.current_timestep()

        frame = copy.deepcopy(frame)
        if isinstance(frame, GpsPoint):
            frame.x -= self.position[0]
            frame.y -= self.position[1]
            frame.altitude -= self.position[2]

        self.visited_frames.append(
            Frame(
                frame=frame,
                time_delta=0,
                position_change=np.zeros(3),
                current_position=position,
            )
        )

        return self.visited_frames

    def step(self, position_change: np.ndarray, max_error: float | None = None) -> Tuple[
        ImuData | GpsPoint,
        np.ndarray,
        bool
    ]:
        """This model takes in the current agent position, returns the current frame data as well as position offset that the agent should take at this step to get to the correct position
        """
        self.position += position_change

        current_frame, true_position, terminated = self.episode.step()

        if max_error and np.linalg.norm(true_position - self.position) > max_error:
            self.position = true_position

        # copy the frame so the inplace changes don't cause other issues
        frame = copy.deepcopy(current_frame)

        # Make gps points relative
        if isinstance(frame, GpsPoint):
            frame.x -= self.position[0]
            frame.y -= self.position[1]
            frame.altitude -= self.position[2]

        self.visited_frames.append(
            Frame(
                frame=frame,
                time_delta=self.episode.current_timestep() - self.last_timestep,
                position_change=position_change,
                current_position=self.position,
            )
        )
        self.visited_frames = self.visited_frames[-self.seq_len:]

        self.last_timestep = self.episode.current_timestep()

        return self.visited_frames, true_position - self.position, terminated
