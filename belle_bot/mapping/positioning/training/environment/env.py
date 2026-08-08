import numpy as np

import houston_api_client
from belle_bot.mapping.positioning.training.environment.episode import Episode
from belle_bot.mapping.positioning.training.models import GpsPoint

houston_client = houston_api_client.Client(base_url="http://localhost:5173/")


class Environment:

    def __init__(self, replay_id, seq_len, random_subsample=False):
        self.seq_len = seq_len
        self.random_subsample = random_subsample
        self.current_episode: Episode | None = None

        # todo store the last item frame/position
        self.last_timestep = None
        self.episode_frames = []

        self.current_episode = Episode(replay_id, random_subsample=self.random_subsample)
        if len(self.current_episode.events) == 0:
            print(f"No events in replay: {replay_id}")

        self.episode_frames = []
        self.last_timestep = self.current_episode.events[self.current_episode.current_step_idx][0].timestamp

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
