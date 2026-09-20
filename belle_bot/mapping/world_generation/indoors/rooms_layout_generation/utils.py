from typing import Optional
import numpy as np


def get_edge_overlap(line_a: np.ndarray, line_b: np.ndarray) -> Optional[np.ndarray]:
    """
    Checks if two lines are collinear and overlap.
    Assumes lines are either horizontal or vertical.

    Args:
        line_a: A numpy array of [x1, y1, x2, y2].
        line_b: A numpy array of [x1, y1, x2, y2].

    Returns:
        A numpy array representing the overlapping segment [x1, y1, x2, y2],
        or None if there is no overlap.
    """
    x1_a, y1_a, x2_a, y2_a = line_a
    x1_b, y1_b, x2_b, y2_b = line_b

    # Check if lines are collinear and overlap
    # They are either both horizontal or both vertical if they are room edges from the same type (e.g. top and bottom)

    # Horizontal lines
    if y1_a == y2_a == y1_b == y2_b:
        # Sort x coordinates
        min_a, max_a = min(x1_a, x2_a), max(x1_a, x2_a)
        min_b, max_b = min(x1_b, x2_b), max(x1_b, x2_b)

        overlap_min = max(min_a, min_b)
        overlap_max = min(max_a, max_b)

        if overlap_min < overlap_max:
            return np.array([overlap_min, y1_a, overlap_max, y1_a])

    # Vertical lines
    if x1_a == x2_a == x1_b == x2_b:
        # Sort y coordinates
        min_a, max_a = min(y1_a, y2_a), max(y1_a, y2_a)
        min_b, max_b = min(y1_b, y2_b), max(y1_b, y2_b)

        overlap_min = max(min_a, min_b)
        overlap_max = min(max_a, max_b)

        if overlap_min < overlap_max:
            return np.array([x1_a, overlap_min, x1_a, overlap_max])

    return None


def get_line_length(line: np.ndarray) -> float:
    """
    Calculates the length of a line segment.

    Args:
        line: A numpy array of [x1, y1, x2, y2].

    Returns:
        The Euclidean length of the segment.
    """
    x1, y1, x2, y2 = line
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
