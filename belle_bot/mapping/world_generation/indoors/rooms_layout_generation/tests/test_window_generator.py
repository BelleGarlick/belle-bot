import unittest
import numpy as np
from belle_bot.mapping.room_generation.models import Point, RoomDefinition
from belle_bot.mapping.room_generation.window_generator import (
    get_window_candidates, populate_windows, subtract_segment
)

class TestWindowGenerator(unittest.TestCase):
    def test_subtract_segment_horizontal(self):
        full = np.array([0, 0, 10, 0])
        shared = np.array([2, 0, 5, 0])
        remaining = subtract_segment(full, shared)
        self.assertEqual(len(remaining), 2)
        np.testing.assert_array_equal(remaining[0], np.array([0, 0, 2, 0]))
        np.testing.assert_array_equal(remaining[1], np.array([5, 0, 10, 0]))

    def test_get_window_candidates(self):
        room1 = RoomDefinition(
            room_id="1", tl=Point(0, 0), br=Point(10, 10), doors=[], windows=[]
        )
        room2 = RoomDefinition(
            room_id="2", tl=Point(10, 0), br=Point(20, 10), doors=[], windows=[]
        )
        rooms = [room1, room2]
        room_graph = {"1": {"2": [np.array([10, 0, 10, 10])]}, "2": {"1": []}}
        
        candidates = get_window_candidates(rooms, room_graph)
        self.assertIn("1", candidates)
        # Room 1 right edge is shared, so it should be empty
        self.assertEqual(len(candidates["1"]["right"]), 0)
        # Top, bottom, left should have 1 segment each
        self.assertEqual(len(candidates["1"]["top"]), 1)
        self.assertEqual(len(candidates["1"]["bottom"]), 1)
        self.assertEqual(len(candidates["1"]["left"]), 1)

    def test_populate_windows(self):
        room1 = RoomDefinition(
            room_id="1", tl=Point(0, 0), br=Point(10, 10), doors=[], windows=[]
        )
        window_candidates = {"1": {"top": [np.array([0, 0, 10, 0])]}}
        populate_windows([room1], window_candidates)
        self.assertEqual(len(room1.windows), 1)
        self.assertEqual(room1.windows[0].position.x, 5.0)
        self.assertEqual(room1.windows[0].position.y, 0.0)

if __name__ == '__main__':
    unittest.main()
