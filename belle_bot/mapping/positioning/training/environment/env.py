import math
import random

import numpy as np

from belle_bot.mapping.positioning.training.environment.episode import Episode
from belle_bot.mapping.positioning.training.models import GpsPoint


class Environment:

    def __init__(self, seq_len, random_subsample=False):
        # todo in future have it use a webdataset for a given track
        self.replay_files = [
            "87d554ee-3d54-4440-85cb-756f0c234067.txt",
            "14240eda-3dfc-48e0-921b-c22397ca0981.txt",
            "5ba31b5c-b457-4c80-9dc4-2f42d9f61a9f.txt",
            "5e6e7499-d18b-4c54-b0e1-cf91033b9331.txt",
            "c6d2c516-252c-4844-9743-40a293813ebe.txt",
            "08487f87-0dfe-44f3-9855-27ad04519266.txt",
            "e15996e1-0808-4cde-968b-2a464be592cb.txt",
            "e69aa899-b321-43ff-bee2-470714593662.txt",
            "ef45f299-e989-494c-af8a-d938b7300138.txt"
        ]

        self.seq_len = seq_len
        self.random_subsample = random_subsample
        self.current_episode: Episode | None = None
        self.episode_index = -1

        # todo store the last item frame/position
        self.last_timestep = None
        self.episode_frames = []

    def __len__(self):
        return len(self.replay_files)

    def reset(self) -> np.ndarray:
        self.episode_index += random.randint(0, len(self.replay_files) - 1)
        self.episode_index %= len(self.replay_files)

        self.current_episode = Episode(self.replay_files[self.episode_index], random_subsample=self.random_subsample)

        self.episode_frames = []
        position = self.current_episode.reset()
        self.last_timestep = self.current_episode.events[self.current_episode.current_step_idx][0].timestamp

        return position

    def step(self, agent_position: np.ndarray):
        """This model takes in the current agent position, returns the current frame data as well as position offset that the agent should take at this step to get to the correct position
        """
        current_frame, true_position, terminated = self.current_episode.step()

        # copy the frame so the inplace changes don't cause other issues
        frame = current_frame.copy()

        # Calculate the position change
        position_change = true_position - agent_position

        # Make the frame timestamp relative
        frame_timestamp = frame.timestamp
        frame.timestamp = frame_timestamp - self.last_timestep
        self.last_timestep = frame_timestamp

        # Make gps points relative
        if isinstance(frame, GpsPoint):
            frame.x -= agent_position[0]
            frame.y -= agent_position[1]
            frame.altitude -= agent_position[2]

        self.episode_frames.append(frame)
        self.episode_frames = self.episode_frames[-self.seq_len:]

        return self.episode_frames, position_change, terminated
