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
    prev_position_change: np.ndarray # the previous one taken by the agent
    position: np.ndarray
    prev_position: np.ndarray
    true_position: np.ndarray


class Environment:

    def __init__(self, episode: Episode, seq_len: int):
        self.episode = episode
        self.seq_len = seq_len

        self.position: np.ndarray = np.zeros(3)

        self.visited_frames: list[Frame] = []

        self.last_timestep = 0

    def reset(self) -> list[Frame]:
        last_position = self.position.copy()

        step_data = self.episode.step()
        self.position = self.episode.current_position()
        self.last_timestep = self.episode.current_timestep()

        frame = copy.deepcopy(step_data.frame)
        if isinstance(frame, GpsPoint):
            frame.x -= last_position[0]
            frame.y -= last_position[1]
            frame.altitude -= last_position[2]

        self.visited_frames.append(
            Frame(
                frame=frame,
                time_delta=step_data.delta_time,
                position_change=np.zeros(3),
                position=self.position,
                true_position=step_data.new_position,
                prev_position=last_position,
                prev_position_change=np.zeros(3),
            )
        )

        return self.visited_frames

    def step(self, position_change: np.ndarray, max_error: float | None = None) -> Tuple[list[Frame], bool]:
        """This model takes in the current agent position, returns the current frame data as well as position offset that the agent should take at this step to get to the correct position
        """
        old_position = np.array(self.position)
        step_data = self.episode.step()

        self.position += position_change
        if max_error and np.linalg.norm(step_data.new_position - self.position) > max_error:
            self.position = step_data.last_position

        # copy the frame so the inplace changes don't cause other issues
        frame = copy.deepcopy(step_data.frame)

        # Make gps points relative
        if isinstance(frame, GpsPoint):
            frame.x -= self.position[0]
            frame.y -= self.position[1]
            frame.altitude -= self.position[2]

        self.visited_frames.append(
            Frame(
                frame=frame,
                time_delta=self.episode.current_timestep() - self.last_timestep,
                position_change=np.array(step_data.new_position - self.position),
                true_position=np.array(step_data.new_position),
                position=np.array(self.position),
                prev_position=old_position,
                prev_position_change=np.array(
                    self.visited_frames[-1].position_change
                ),
            )
        )
        self.visited_frames = self.visited_frames[-self.seq_len:]

        self.last_timestep = self.episode.current_timestep()

        return self.visited_frames, step_data.terminated
