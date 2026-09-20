import uuid

import numpy as np

from belle_bot.mapping.world_generation.indoors.room_generation.tiles import RoomGenerationGrid, Cell, TILE_MAP
from belle_bot.mapping.world_generation.indoors.rooms_layout_generation.models import RoomDefinition
from belle_bot.mapping.world_generation.indoors.rooms_layout_generation.world_generator import (
    render_rooms_layout,
    generate_rooms_layout
)

def populate_room(room: RoomDefinition):
    print(room)

    grid_width = round(room.width) + 2
    grid_height = round(room.height) + 2

    grid = RoomGenerationGrid(rows=grid_height, cols=grid_width)

    room_left = room.tl.x
    room_bottom = room.br.y

    for door in room.doors:
        door_rel_x = int(((door.position.x - room_left) / room.width) * (grid_width - 2)) + 1
        door_rel_y = int(((door.position.y - room_bottom) / room.height) * (grid_height - 2)) + 1

        grid[door_rel_x, door_rel_y] = Cell(
            possible_cells=[TILE_MAP['door']],
            collapsed=True,
            entity_id=str(uuid.uuid4())
        )

    for window in room.windows:
        window_rel_x = int(((window.position.x - room_left) / room.width) * grid_width)
        window_rel_y = int(((window.position.y - room_bottom) / room.height) * grid_height)

        grid[window_rel_x, window_rel_y] = Cell(
            possible_cells=[TILE_MAP['window']],
            collapsed=True,
            entity_id=str(uuid.uuid4())
        )

    grid.wave_collapse()
    grid.print()

    # breakpoint()

    # plt.plot(room.top_edge[0::2], room.top_edge[1::2], c='red')
    # plt.plot(room.left_edge[0::2], room.left_edge[1::2], c='red')
    # plt.plot(room.right_edge[0::2], room.right_edge[1::2], c='red')
    # plt.plot(room.bottom_edge[0::2], room.bottom_edge[1::2], c='red')
    #
    # for door in room.doors:
    #     plt.scatter(
    #         [door.position.x],
    #         [door.position.y],
    #     )
    # for window in room.windows:
    #     plt.scatter(
    #         [window.position.x],
    #         [window.position.y],
    #     )
    #
    # plt.show()


if __name__ == "__main__":
    np.random.seed(0)

    rooms = generate_rooms_layout(
        width=30,
        height=10
    )

    for room in rooms:
        populate_room(room)

    total_doors = sum(len(room.doors) for room in rooms) // 2
    total_windows = sum(len(room.windows) for room in rooms)
    print(f"Total rooms: {len(rooms)}")
    print(f"Total doors placed: {total_doors}")
    print(f"Total windows placed: {total_windows}")

    render_rooms_layout(rooms)
