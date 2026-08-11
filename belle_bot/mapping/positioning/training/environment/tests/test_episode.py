import unittest
from unittest.mock import patch

import numpy as np

from belle_bot.mapping.positioning.training.environment.episode import (
    Episode,
    _create_event_pairs,
    _get_initial_gps_pos,
    _parse_events,
    _subsample_events,
    calculate_catmull_rom_segment,
)

MODULE_PATH = "belle_bot.mapping.positioning.training.environment.episode"


class MockGpsPoint:
    def __init__(self, timestamp, x=0.0, y=0.0, altitude=0.0, name=None):
        self.timestamp = timestamp
        self.x = x
        self.y = y
        self.altitude = altitude
        self.name = name

    @classmethod
    def from_data(cls, timestamp, data):
        return cls(
            timestamp=timestamp,
            x=data.get("x", 0.0),
            y=data.get("y", 0.0),
            altitude=data.get("altitude", 0.0),
        )

    def numpy(self):
        return np.array([self.x, self.y, self.altitude])


class MockImuData:
    def __init__(self, timestamp, name=None):
        self.timestamp = timestamp
        self.name = name

    @classmethod
    def from_data(cls, timestamp, data):
        return cls(timestamp=timestamp)


@patch(f"{MODULE_PATH}.GpsPoint", MockGpsPoint)
@patch(f"{MODULE_PATH}.ImuData", MockImuData)
class TestEventParsingAndUtilities(unittest.TestCase):

    @patch(f"{MODULE_PATH}.replays.get_replay_file")
    def test_parse_events(self, mock_get_replay):
        csv_data = "\n".join([
            'sensors/gps,10.0,{"has_fix": "True", "x": 1.0, "y": 2.0, "altitude": 3.0}',
            'sensors/gps,11.0,{"has_fix": "False", "x": 0.0, "y": 0.0}',
            'sensors/imu,12.0,{"accel": [0, 0, 9.8]}',
            'sensors/unknown,13.0,{}'
        ])
        mock_get_replay.return_value = csv_data

        events = _parse_events("test_replay_id")

        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], MockGpsPoint)
        self.assertEqual(events[0].timestamp, 10.0)
        self.assertEqual(events[0].name, "0")

        self.assertIsInstance(events[1], MockImuData)
        self.assertEqual(events[1].timestamp, 12.0)
        self.assertEqual(events[1].name, "2")

    def test_subsample_events(self):
        events = list(range(10))
        subsampled = _subsample_events(events)

        half_len = len(events) // 2
        self.assertEqual(len(subsampled), half_len)

    def test_create_event_pairs(self):
        g1 = MockGpsPoint(10.0)
        g2 = MockGpsPoint(20.0)
        g3 = MockGpsPoint(30.0)
        g4 = MockGpsPoint(40.0)
        i1 = MockImuData(25.0)

        events = [g1, g2, i1, g3, g4]
        pairs = _create_event_pairs(events)

        self.assertEqual(len(pairs), 3)
        item, gps0, gps1, gps2, gps3 = pairs[1]

        self.assertIs(item, i1)
        self.assertIs(gps0, g1)
        self.assertIs(gps1, g2)
        self.assertIs(gps2, g3)
        self.assertIs(gps3, g4)

    def test_get_initial_gps_pos(self):
        g1 = MockGpsPoint(10.0)
        g2 = MockGpsPoint(20.0)
        i1 = MockImuData(15.0)

        events = [i1, g1, i1, g2]
        result = _get_initial_gps_pos(events)

        self.assertIsNotNone(result)
        item, idx = result
        self.assertIs(item, g2)
        self.assertEqual(idx, 3)

    def test_get_initial_gps_pos_insufficient_gps(self):
        events = [MockImuData(1.0), MockGpsPoint(2.0)]
        self.assertIsNone(_get_initial_gps_pos(events))

    def test_calculate_catmull_rom_segment(self):
        p0 = np.array([0.0, 0.0])
        p1 = np.array([1.0, 1.0])
        p2 = np.array([2.0, 1.0])
        p3 = np.array([3.0, 0.0])

        res_start = calculate_catmull_rom_segment(p0, p1, p2, p3, t=0.0)
        res_end = calculate_catmull_rom_segment(p0, p1, p2, p3, t=1.0)

        np.testing.assert_array_almost_equal(res_start, p1)
        np.testing.assert_array_almost_equal(res_end, p2)


@patch(f"{MODULE_PATH}.GpsPoint", MockGpsPoint)
@patch(f"{MODULE_PATH}.ImuData", MockImuData)
class TestEpisode(unittest.TestCase):

    def setUp(self):
        self.g0 = MockGpsPoint(10.0, x=0.0, y=0.0, altitude=0.0)
        self.g1 = MockGpsPoint(20.0, x=10.0, y=0.0, altitude=5.0)
        self.g2 = MockGpsPoint(40.0, x=30.0, y=0.0, altitude=15.0)
        self.g3 = MockGpsPoint(50.0, x=40.0, y=0.0, altitude=20.0)

        self.imu = MockImuData(30.0)

    @patch(f"{MODULE_PATH}._parse_events")
    def test_episode_initialization_and_position(self, mock_parse_events):
        mock_parse_events.return_value = [self.g0, self.g1, self.imu, self.g2, self.g3]

        episode = Episode("dummy_path", random_subsample=False)

        self.assertEqual(len(episode.events), 3)
        self.assertEqual(episode.current_timestep(), 20.0)
        self.assertIs(episode.get_current_frame(), self.g1)

        pos = episode.current_position()

        self.assertAlmostEqual(pos[2], 5)
        self.assertIsInstance(pos, np.ndarray)
        self.assertEqual(pos.shape, (3,))

    @patch(f"{MODULE_PATH}._parse_events")
    def test_episode_gps_point_position(self, mock_parse_events):
        # Position of a GpsPoint direct returning numpy array
        gps_point_item = MockGpsPoint(30.0, x=20.0, y=5.0, altitude=10.0)
        mock_parse_events.return_value = [self.g0, self.g1, gps_point_item, self.g2, self.g3]

        episode = Episode("dummy_path")

        pos = episode.calculate_position(1)
        np.testing.assert_array_equal(pos, np.array([20.0, 5.0, 10.0]))

    @patch(f"{MODULE_PATH}._parse_events")
    def test_episode_step(self, mock_parse_events):
        i1 = MockImuData(25.0)
        i2 = MockImuData(35.0)

        mock_parse_events.return_value = [self.g0, self.g1, i1, i2, self.g2, self.g3]
        episode = Episode("dummy_path")

        self.assertEqual(len(episode.events), 4)  # Pairs for g1, i1, i2 and g2

        # Step 1
        item, pos, done = episode.step()
        self.assertIs(item, self.g1)
        self.assertFalse(done)
        self.assertEqual(episode.current_step_idx, 1)

        item, pos, done = episode.step()
        self.assertIs(item, i1)
        self.assertFalse(done)
        self.assertEqual(episode.current_step_idx, 2)

        item, pos, done = episode.step()
        self.assertIs(item, i2)
        self.assertTrue(done)
        self.assertEqual(episode.current_step_idx, 3)
