import unittest
import numpy as np
from belle_bot.mapping.room_generation.utils import get_edge_overlap, get_line_length

class TestUtils(unittest.TestCase):
    def test_get_edge_overlap_horizontal(self):
        line_a = np.array([0, 0, 10, 0])
        line_b = np.array([5, 0, 15, 0])
        overlap = get_edge_overlap(line_a, line_b)
        self.assertTrue(np.array_equal(overlap, np.array([5, 0, 10, 0])))

    def test_get_edge_overlap_no_overlap(self):
        line_a = np.array([0, 0, 5, 0])
        line_b = np.array([6, 0, 10, 0])
        overlap = get_edge_overlap(line_a, line_b)
        self.assertIsNone(overlap)

    def test_get_line_length(self):
        line = np.array([0, 0, 3, 4])
        self.assertEqual(get_line_length(line), 5.0)

if __name__ == '__main__':
    unittest.main()
