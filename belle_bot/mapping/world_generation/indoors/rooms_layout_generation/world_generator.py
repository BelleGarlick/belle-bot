from matplotlib import pyplot as plt

from belle_bot.mapping.world_generation.indoors.rooms_layout_generation.rooms_layout_generator import generate_room_layout
from belle_bot.mapping.world_generation.indoors.rooms_layout_generation.graph_utils import create_room_graph, drop_rooms, create_door_graph, populate_doors
from belle_bot.mapping.world_generation.indoors.rooms_layout_generation.window_generator import get_window_candidates, populate_windows
from belle_bot.mapping.world_generation.indoors.rooms_layout_generation.models import RoomDefinition

def generate_rooms_layout(
    width: float = 100,
    height: float = 60,
    dropped_room_count: int = 2,
    extra_door_probability: float = 0.2
) -> list[RoomDefinition]:
    """
    Generates a complete world layout with rooms, doors, and windows.

    Args:
        width: The total width of the building.
        height: The total height of the building.
        dropped_room_count: Number of rooms to remove for non-rectangular shape.
        extra_door_probability: Probability of adding extra doors between rooms.

    Returns:
        A list of populated RoomDefinition objects.
    """
    # Create rooms via tree-bisection a grid
    rooms = generate_room_layout(width, height)

    # Create the room graph based on overlapping edges
    room_graph = create_room_graph(rooms)

    # Drop rooms so we don't have a plain rectangle shape
    rooms, room_graph = drop_rooms(rooms, room_graph, dropped_room_count=dropped_room_count)

    # Create the graph of rooms that has a door between
    door_graph = create_door_graph(room_graph, extra_door_probability=extra_door_probability)

    # Window candidates
    window_candidates = get_window_candidates(rooms, room_graph)

    # Populate doors and windows in room definitions
    populate_doors(rooms, door_graph, room_graph)
    populate_windows(rooms, window_candidates)

    return rooms

def render_rooms_layout(rooms: list[RoomDefinition]) -> None:
    """
    Renders the generated world using matplotlib.

    Args:
        rooms: The list of RoomDefinition objects to render.
    """
    # Create the room graph again for visualization purposes (door connections)
    # Alternatively, we could pass door_graph and window_candidates if we wanted to avoid re-calculation,
    # but the prompt asked for the function to return room definitions, so we'll use those.
    room_graph = create_room_graph(rooms)
    window_candidates = get_window_candidates(rooms, room_graph)
    
    # We need to know which rooms have doors between them. 
    # We can reconstruct this from the doors in the room definitions.
    door_connections = []
    processed_doors = set()
    for room in rooms:
        for door in room.doors:
            if door.door_id not in processed_doors:
                processed_doors.add(door.door_id)
                if len(door.room_ids) == 2:
                    door_connections.append(list(door.room_ids))

    room_map = {room.room_id: room for room in rooms}

    for leaf in rooms:
        plt.plot(leaf.top_edge[0::2], leaf.top_edge[1::2], 'grey')
        plt.plot(leaf.left_edge[0::2], leaf.left_edge[1::2], 'grey')
        plt.plot(leaf.right_edge[0::2], leaf.right_edge[1::2], 'grey')
        plt.plot(leaf.bottom_edge[0::2], leaf.bottom_edge[1::2], 'grey')

        plt.scatter(leaf.center[0:1], leaf.center[1:2], color='blue', s=10)

        # Render doors
        for door in leaf.doors:
            plt.scatter(door.position.x, door.position.y, color='red', marker='s', s=30, zorder=3)

        # Render windows
        for window in leaf.windows:
            plt.scatter(window.position.x, window.position.y, color='gold', marker='*', s=30, zorder=3)

        # Render objects
        for obj in leaf.objects:
            plt.scatter(
                obj['position'].x,
                obj['position'].y,
                color='green',
                marker='o',
                s=15,
                zorder=2,
                alpha=0.6
            )
            plt.text(
                obj['position'].x,
                obj['position'].y,
                obj['tile']['name'],
                fontsize=6,
                ha='center',
                va='bottom'
            )

    for connection in door_connections:
        room_i, room_j = connection
        if room_i in room_map and room_j in room_map:
            plt.plot(
                [room_map[room_i].center[0], room_map[room_j].center[0]],
                [room_map[room_i].center[1], room_map[room_j].center[1]],
                'k--', alpha=0.5
            )

    for room_id, edges in window_candidates.items():
        for edge_type, segments in edges.items():
            for segment in segments:
                plt.plot(segment[0::2], segment[1::2], 'c', linewidth=3, alpha=0.5)

    plt.axis('equal')
    plt.show()
