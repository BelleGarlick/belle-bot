import uuid
from dataclasses import dataclass

import numpy as np

from belle_bot.mapping.world_generation.inside_generation.rooms_layout_generation.models import Point, RoomDefinition


@dataclass
class Node:
    """
    Represents a rectangular region in the building layout, which can be subdivided.
    """
    tl: Point
    br: Point

    left: Node | None = None
    right: Node | None = None

    @property
    def width(self) -> float:
        return self.br.x - self.tl.x

    @property
    def height(self) -> float:
        return self.br.y - self.tl.y

    def __hash__(self) -> int:
        return hash((self.tl, self.br))

    def get_leafs(self) -> set['Node']:
        """
        Recursively finds all leaf nodes (nodes that haven't been split further).

        Returns:
            A set of leaf Node objects.
        """
        leafs = set()

        if self.left is not None:
            leafs.update(self.left.get_leafs())

        if self.right is not None:
            leafs.update(self.right.get_leafs())

        if self.left is None and self.right is None:
            leafs.add(self)

        return leafs

    def perform_split(self) -> None:
        """
        Splits the current node into two child nodes (left and right).
        The split is performed either horizontally or vertically based on the node's dimensions
        and a random factor.
        """
        split = float(np.random.normal(0.5, 0.1))
        # sample based on longest side

        if np.random.normal(0.5, 0.1) * (self.width + self.height) > self.width:
            # split vert
            split = self.tl.y + (self.br.y - self.tl.y) * split
            self.left = Node(
                tl=self.tl,
                br=Point(x=self.br.x, y=split),
            )
            self.right = Node(
                tl=Point(x=self.tl.x, y=split),
                br=self.br,
            )

        else:
            # split horz
            split = self.tl.x + (self.br.x - self.tl.x) * split
            self.left = Node(
                tl=self.tl,
                br=Point(x=split, y=self.br.y),
            )
            self.right = Node(
                tl=Point(x=split, y=self.tl.y),
                br=self.br,
            )


def generate_room_layout(width: float, height: float) -> list[RoomDefinition]:
    """
    Generates a building layout by recursively bisecting a grid into smaller rooms.

    Args:
        width: The total width of the building.
        height: The total height of the building.

    Returns:
        A list of RoomDefinition objects representing the generated rooms.
    """
    building = Node(
        tl=Point(0, 0),
        br=Point(width, height),
    )

    # todo scale rooms with size of building
    for _ in range(8):
        leafs = [*building.get_leafs()]
        probs = [np.random.uniform() * min(leaf.width, leaf.height) for leaf in leafs]
        idx = np.argmax(probs)
        leafs[idx].perform_split()

    return [
        RoomDefinition(
            room_id=str(uuid.uuid4()),
            tl=leaf.tl,
            br=leaf.br,
            doors=[],
            windows=[]
        )
        for leaf in building.get_leafs()
    ]
