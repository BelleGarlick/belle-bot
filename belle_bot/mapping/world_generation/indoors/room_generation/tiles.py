import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from typing import Optional
import numpy as np

@dataclass
class Cell:
    possible_cells: list[dict]
    collapsed: bool = False
    entity_id: Optional[str] = None

    def __len__(self):
        return len(self.possible_cells)

    def __iter__(self):
        return iter(self.possible_cells)

    def __getitem__(self, item):
        return self.possible_cells[item]

    @property
    def is_wall(self):
        return len(self) == 1 and self[-1]['id'] == 'wall'


class RoomGenerationGrid:

    def __init__(self, rows=1, cols=1):
        self.rows = rows
        self.cols = cols

        # Create grid of all items
        spawnable_tiles = [tile for tile in tiles if tile["spawnable"]]
        self.grid: list[list[Cell]] = [
            [
                Cell([*spawnable_tiles]) for _ in range(cols)
            ] for _ in range(rows)
        ]

        # Collapse the walls
        collapsed_wall = Cell([TILE_MAP["wall"]], entity_id="Wall", collapsed=True)
        for col in range(cols):
            self.grid[0][col] = collapsed_wall
            self.grid[-1][col] = collapsed_wall

        for row in range(rows):
            self.grid[row][0] = collapsed_wall
            self.grid[row][-1] = collapsed_wall

    def __setitem__(self, key, value):
        self.grid[key[0] % self.rows][key[1] % self.cols] = value

    def __getitem__(self, item):
        return self.grid[item[0] % self.rows][item[1] % self.cols]

    def print(self):
        for row in range(self.rows):
            row_string = ""
            for col in range(self.cols):
                current_cell = self[row, col]
                current_item_count = len(current_cell)

                if len(current_cell) == 1:
                    row_string += current_cell[0]['name'].rjust(12)

                else:
                    row_string += str(current_item_count).rjust(9) + " | "

            print(row_string)

        print()

    def is_next_to_wall(self, row, col):
        return (
            self[row, col - 1].is_wall
            or self[row, col + 1].is_wall
            or self[row - 1, col].is_wall
            or self[row + 1, col].is_wall
        )

    def __reduce_once(self) -> int:
        collapsed_cells = 0

        for row, col in self:
            current_cell = self[row, col]
            current_item_count = len(current_cell)

            if current_cell.collapsed:
                continue

            # Filter for ones that aren't next to a wall
            if not self.is_next_to_wall(row, col):
                current_cell = [x for x in current_cell if not x['next_to_wall']]

            # Filter out items which have no viable permutations that they can be placed in
            current_cell = [
                cell
                for cell in current_cell
                if len(get_permutations(self, cell, (row, col))) > 0
            ]

            # todo have a way to specify invalid neighbours. eg toilet not next to a sofa

            if len(current_cell) != current_item_count:
                collapsed_cells += 1

            self[row, col].possible_cells = current_cell

        return collapsed_cells

    def reduce(self):
        for idx in range(100):
            collapsed_cells = self.__reduce_once()

            if collapsed_cells == 0:
                break

            if idx == 99:
                raise Exception("Max collapse iteration reached")

    def collapse_cell(self):
        for row, col in self:
            if not self[row, col].collapsed:
                items = self[row, col]

                # Create weighted cells
                weights = [np.random.uniform(0, 1) * item['weight'] for item in items]
                chosen_item = items[np.argmax(weights)]

                # Get the layout permutation
                permutations = get_permutations(self, chosen_item, (row, col))
                permutations_rand = [np.random.random() for _ in permutations]
                permutation = permutations[np.argmax(permutations_rand)]

                entity_id = str(uuid.uuid4())
                for cell in permutation:
                    r, c = cell
                    self[r, c] = Cell([chosen_item], entity_id=entity_id, collapsed=True)

                return False

        return True

    def __iter__(self):
        self.__iter_item = 0, 0

        return self

    def __next__(self):
        row, col = self.__iter_item
        col += 1

        if col >= self.cols:
            col = 0
            row += 1

        if row >= self.rows:
            raise StopIteration

        self.__iter_item = row, col

        return row, col

    def wave_collapse(self):
        self.reduce()

        while True:
            completed = self.collapse_cell()
            self.reduce()

            if completed:
                break


tile_dir = Path(__file__).parent / "tiles"


def load_tiles():
    tiles = []
    for file in tile_dir.glob("*.json"):
        with file.open() as f:
            tiles.append(
                json.load(f)
            )

    return tiles


def populate_tile_data(tiles: list[dict]) -> list[dict]:
    return [
        {
            "spawnable": False,
            "next_to_wall": False,
            "weight": 1,
            "layouts": ["1u"],
            **tile_data
        }
        for tile_data in tiles
    ]


def get_permutations(grid: RoomGenerationGrid, tile, cell_idx):
    row, col = cell_idx
    permutated_cells = []

    # Iterate through the layouts to find all permutations of cells we want to try and populate
    for layout in tile['layouts']:
        if layout == "2u":
            permutated_cells.append([(row, col), (row, col + 1)])
            permutated_cells.append([(row, col), (row, col - 1)])

            permutated_cells.append([(row, col), (row + 1, col )])
            permutated_cells.append([(row, col), (row - 1, col )])

        elif layout == "3u":
            permutated_cells.append([(row, col), (row + 1, col), (row + 2, col)])
            permutated_cells.append([(row, col), (row - 1, col), (row - 2, col)])
            permutated_cells.append([(row, col), (row - 1, col), (row + 1, col)])

            permutated_cells.append([(row, col), (row, col + 1), (row, col + 2)])
            permutated_cells.append([(row, col), (row, col - 1), (row, col - 2)])
            permutated_cells.append([(row, col), (row, col - 1), (row, col + 1)])

        elif layout == "1u":
            permutated_cells.append([(row, col)])

        elif layout == "2x2":
            # todo deal with rotations a little smarter
            permutated_cells.append([
                (row, col), (row, col + 1),
                (row + 1, col), (row + 1, col + 1),
            ])
            permutated_cells.append([
                (row, col), (row, col - 1),
                (row + 1, col), (row + 1, col - 1),
            ])
            permutated_cells.append([
                (row, col), (row, col + 1),
                (row - 1, col), (row - 1, col + 1),
            ])
            permutated_cells.append([
                (row, col), (row, col - 1),
                (row - 1, col), (row - 1, col - 1),
            ])

        else:
            breakpoint()

    # todo potentially apply offsets and rotations that are valid

    # Iterate through the permutations to find the permutations that are valid
    cells = []
    for cell_permutations in permutated_cells:
        valid = True
        for r, c in cell_permutations:
            if grid[r, c].collapsed:
                valid = False
                break
        if valid:
            cells.append(cell_permutations)

        # todo filter out pemutations that have a dodgy neightbou

    return cells


tiles = load_tiles()
tiles = populate_tile_data(tiles)
TILE_MAP = {x['id']: x for x in tiles}


if __name__ == "__main__":
    grid = RoomGenerationGrid(rows=5, cols=8)
    grid.print()

    grid.wave_collapse()
    grid.print()
