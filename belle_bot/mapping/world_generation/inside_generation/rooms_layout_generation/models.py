from typing import List
from dataclasses import dataclass

import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


@dataclass
class Point:
    """Represents a 2D point with x and y coordinates."""
    x: float
    y: float

    def __hash__(self):
        return hash((self.x, self.y))


@dataclass
class Door:
    """Represents a door at a specific position."""
    door_id: str
    position: Point

    room_ids: set[str]


@dataclass
class Window:
    """Represents a window at a specific position."""
    position: Point


@dataclass
class RoomDefinition:
    """
    Defines a room with its boundaries, doors, and windows.
    """
    room_id: str

    tl: Point
    br: Point

    doors: List[Door]
    windows: List[Window]

    @property
    def center(self) -> np.ndarray:
        """Returns the center point of the room as a numpy array [x, y]."""
        return np.array([
            (self.tl.x + self.br.x) / 2,
            (self.tl.y + self.br.y) / 2,
        ])

    @property
    def top_edge(self) -> np.ndarray:
        return np.array([self.tl.x, self.tl.y, self.br.x, self.tl.y])

    @property
    def bottom_edge(self) -> np.ndarray:
        return np.array([self.tl.x, self.br.y, self.br.x, self.br.y])

    @property
    def left_edge(self) -> np.ndarray:
        return np.array([self.tl.x, self.tl.y, self.tl.x, self.br.y])

    @property
    def right_edge(self) -> np.ndarray:
        return np.array([self.br.x, self.tl.y, self.br.x, self.br.y])
