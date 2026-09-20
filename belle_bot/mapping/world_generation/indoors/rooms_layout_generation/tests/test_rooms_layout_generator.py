import unittest
from belle_bot.mapping.room_generation.rooms_layout_generator import generate_room_layout, Node
from belle_bot.mapping.room_generation.models import Point

class TestRoomsLayoutGenerator(unittest.TestCase):
    def test_node_dimensions(self):
        node = Node(tl=Point(0, 0), br=Point(10, 20))
        self.assertEqual(node.width, 10)
        self.assertEqual(node.height, 20)

    def test_generate_room_layout(self):
        rooms = generate_room_layout(100, 60)
        self.assertGreater(len(rooms), 0)
        for room in rooms:
            self.assertIsNotNone(room.room_id)
            self.assertGreater(room.br.x, room.tl.x)
            self.assertGreater(room.br.y, room.tl.y)

    def test_perform_split(self):
        node = Node(tl=Point(0, 0), br=Point(100, 100))
        node.perform_split()
        self.assertIsNotNone(node.left)
        self.assertIsNotNone(node.right)
        
if __name__ == '__main__':
    unittest.main()
