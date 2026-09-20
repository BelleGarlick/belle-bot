import unittest
import numpy as np
from belle_bot.mapping.room_generation.models import Point, RoomDefinition

class TestModels(unittest.TestCase):
    def test_point_hash(self):
        p1 = Point(1.0, 2.0)
        p2 = Point(1.0, 2.0)
        self.assertEqual(hash(p1), hash(p2))

    def test_room_definition_center(self):
        room = RoomDefinition(
            room_id="test",
            tl=Point(0, 0),
            br=Point(10, 10),
            doors=[],
            windows=[]
        )
        np.testing.assert_array_equal(room.center, np.array([5.0, 5.0]))

    def test_room_definition_edges(self):
        room = RoomDefinition(
            room_id="test",
            tl=Point(0, 0),
            br=Point(10, 10),
            doors=[],
            windows=[]
        )
        np.testing.assert_array_equal(room.top_edge, np.array([0, 0, 10, 0]))
        np.testing.assert_array_equal(room.bottom_edge, np.array([0, 10, 10, 10]))
        np.testing.assert_array_equal(room.left_edge, np.array([0, 0, 0, 10]))
        np.testing.assert_array_equal(room.right_edge, np.array([10, 0, 10, 10]))

if __name__ == '__main__':
    unittest.main()
