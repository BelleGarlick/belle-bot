import uuid

from belle_bot.mapping.world_generation.indoors.room_generation.tiles import RoomGenerationGrid, Cell, TILE_MAP
from belle_bot.mapping.world_generation.indoors.rooms_layout_generation.models import RoomDefinition, Point
from belle_bot.mapping.world_generation.indoors.rooms_layout_generation.world_generator import (
    generate_rooms_layout
)

def populate_room(room: RoomDefinition):
    grid_width = round(room.width) + 2
    grid_height = round(room.height) + 2

    grid = RoomGenerationGrid(rows=grid_height, cols=grid_width)

    for door in room.doors:
        door_pos_x = (door.position.x - room.tl.x) / room.width
        door_pos_y = (room.tl.y - door.position.y) / room.height

        cell_x = int(round(door_pos_x * (grid_width - 1)))
        cell_y = int(round(door_pos_y * (grid_height - 1)))

        grid[cell_y, cell_x] = Cell(
            possible_cells=[TILE_MAP['door']],
            collapsed=True,
            entity_id=str(uuid.uuid4())
        )

    for window in room.windows:
        window_pos_x = (window.position.x - room.tl.x) / room.width
        window_pos_y = (room.tl.y - window.position.y) / room.height

        cell_x = int(round(window_pos_x * (grid_width - 1)))
        cell_y = int(round(window_pos_y * (grid_height - 1)))

        grid[cell_y, cell_x] = Cell(
            possible_cells=[TILE_MAP['window']],
            collapsed=True,
            entity_id=str(uuid.uuid4())
        )

    grid.wave_collapse()
    grid.print()

    # Store objects in room definition for rendering
    for row in range(grid.rows):
        for col in range(grid.cols):
            cell = grid[row, col]
            if cell.collapsed and len(cell.possible_cells) == 1:
                tile = cell.possible_cells[0]
                if tile['spawnable'] and tile['id'] != 'floor':
                    # Map grid coordinates back to world coordinates
                    world_x = room.tl.x + (col / (grid_width - 1)) * room.width
                    world_y = room.tl.y - (row / (grid_height - 1)) * room.height
                    
                    room.objects.append({
                        'tile': tile,
                        'position': Point(world_x, world_y),
                        'entity_id': cell.entity_id
                    })
    
    print(f"Stored {len(room.objects)} objects for room {room.room_id}")

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

    # render_rooms_layout(rooms)
