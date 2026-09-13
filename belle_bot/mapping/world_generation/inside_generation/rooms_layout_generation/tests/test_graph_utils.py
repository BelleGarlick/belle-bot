import unittest
import numpy as np
from belle_bot.mapping.room_generation.models import Point, RoomDefinition
from belle_bot.mapping.room_generation.graph_utils import (
    get_shared_edges, create_room_graph, is_graph_connected, 
    drop_rooms, create_door_graph, populate_doors
)

class TestGraphUtils(unittest.TestCase):
    def setUp(self):
        self.room1 = RoomDefinition(
            room_id="1", tl=Point(0, 0), br=Point(10, 10), doors=[], windows=[]
        )
        self.room2 = RoomDefinition(
            room_id="2", tl=Point(10, 0), br=Point(20, 10), doors=[], windows=[]
        )
        self.room3 = RoomDefinition(
            room_id="3", tl=Point(0, 10), br=Point(10, 20), doors=[], windows=[]
        )

    def test_get_shared_edges(self):
        edges = get_shared_edges(self.room1, self.room2)
        self.assertEqual(len(edges), 1)
        np.testing.assert_array_equal(edges[0], np.array([10, 0, 10, 10]))

    def test_create_room_graph(self):
        rooms = [self.room1, self.room2]
        graph = create_room_graph(rooms)
        self.assertIn("1", graph)
        self.assertIn("2", graph)
        self.assertIn("2", graph["1"])

    def test_is_graph_connected(self):
        graph = {"1": {"2": []}, "2": {"1": []}}
        self.assertTrue(is_graph_connected(["1", "2"], graph))
        self.assertFalse(is_graph_connected(["1", "2", "3"], graph))

    def test_create_door_graph(self):
        room_graph = {"1": {"2": [np.array([10, 0, 10, 10])]}, "2": {"1": []}}
        door_graph = create_door_graph(room_graph)
        self.assertIn("1", door_graph)
        self.assertIn("2", door_graph["1"])

    def test_populate_doors(self):
        rooms = [self.room1, self.room2]
        room_graph = create_room_graph(rooms)
        door_graph = {"1": {"2"}, "2": {"1"}}
        populate_doors(rooms, door_graph, room_graph)
        self.assertEqual(len(self.room1.doors), 1)
        self.assertEqual(len(self.room2.doors), 1)
        self.assertEqual(self.room1.doors[0].door_id, self.room2.doors[0].door_id)

if __name__ == '__main__':
    unittest.main()
