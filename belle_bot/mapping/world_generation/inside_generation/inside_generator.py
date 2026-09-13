from matplotlib import pyplot as plt

from belle_bot.mapping.world_generation.inside_generation.rooms_layout_generation.models import RoomDefinition
from belle_bot.mapping.world_generation.inside_generation.rooms_layout_generation.world_generator import (
    render_rooms_layout,
    generate_rooms_layout
)

def populate_room(room: RoomDefinition):
    print(room)

    plt.plot(room.top_edge[0::2], room.top_edge[1::2], c='red')
    plt.plot(room.left_edge[0::2], room.left_edge[1::2], c='red')
    plt.plot(room.right_edge[0::2], room.right_edge[1::2], c='red')
    plt.plot(room.bottom_edge[0::2], room.bottom_edge[1::2], c='red')

    for door in room.doors:
        plt.scatter(
            [door.position.x],
            [door.position.y],
        )
    for window in room.windows:
        plt.scatter(
            [window.position.x],
            [window.position.y],
        )

    plt.show()


if __name__ == "__main__":
    rooms = generate_rooms_layout()

    for room in rooms:
        populate_room(room)

    total_doors = sum(len(room.doors) for room in rooms) // 2
    total_windows = sum(len(room.windows) for room in rooms)
    print(f"Total rooms: {len(rooms)}")
    print(f"Total doors placed: {total_doors}")
    print(f"Total windows placed: {total_windows}")

    render_rooms_layout(rooms)
