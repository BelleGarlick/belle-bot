import uuid
import random
import heapq
from typing import Any

import numpy as np

from belle_bot.mapping.world_generation.inside_generation.rooms_layout_generation.models import (
    RoomDefinition,
    Door,
    Point
)
from belle_bot.mapping.world_generation.inside_generation.rooms_layout_generation.utils import (
    get_edge_overlap,
    get_line_length
)


def get_shared_edges(room_a: RoomDefinition, room_b: RoomDefinition, min_edge_length: float = 1.0) -> list[np.ndarray]:
    """
    Identifies shared wall segments between two rooms.

    Args:
        room_a: The first room.
        room_b: The second room.
        min_edge_length: Minimum length for a shared segment to be considered valid.

    Returns:
        A list of numpy arrays representing shared edge segments.
    """
    overlapping_edges = [
        get_edge_overlap(room_a.top_edge, room_b.bottom_edge),
        get_edge_overlap(room_a.left_edge, room_b.right_edge),
        get_edge_overlap(room_a.bottom_edge, room_b.top_edge),
        get_edge_overlap(room_a.right_edge, room_b.left_edge),
    ]

    # Filter edges
    non_null_edges = [x for x in overlapping_edges if x is not None]
    filtered_edges = [edge for edge in non_null_edges if get_line_length(edge) >= min_edge_length]

    return filtered_edges


def create_room_graph(rooms: list[RoomDefinition]) -> dict[str, dict[str, list[np.ndarray]]]:
    """
    Constructs a graph where nodes are room IDs and edges represent shared walls.
    Each edge in the graph contains the segments of the shared walls.

    Args:
        rooms: A list of room definitions.

    Returns:
        A adjacency dictionary representing the room graph.
    """
    graph = {}

    for room_i in range(len(rooms) - 1):
        for room_j in range(room_i + 1, len(rooms)):
            shared_edges = get_shared_edges(rooms[room_i], rooms[room_j], min_edge_length=2)
            if shared_edges:
                room_i_id = rooms[room_i].room_id
                room_j_id = rooms[room_j].room_id

                if room_i_id not in graph:
                    graph[room_i_id] = {}
                if room_j_id not in graph:
                    graph[room_j_id] = {}

                graph[room_i_id][room_j_id] = shared_edges
                graph[room_j_id][room_i_id] = shared_edges

    return graph


def is_graph_connected(room_ids: list[str], room_graph: dict[str, Any]) -> bool:
    """
    Checks if a subset of rooms remains connected within the given room graph.
    Handles both weighted/detailed graphs and simple adjacency set graphs.

    Args:
        room_ids: The list of room IDs to check for connectivity.
        room_graph: The graph structure to traverse.

    Returns:
        True if all rooms in room_ids are connected, False otherwise.
    """
    if not room_ids:
        return True
    
    start_node = room_ids[0]
    visited = {start_node}
    stack = [start_node]
    
    while stack:
        current = stack.pop()
        # Check edges from current to others
        neighbors = []
        if current in room_graph:
            if isinstance(room_graph[current], dict):
                neighbors.extend(room_graph[current].keys())
            else:
                neighbors.extend(room_graph[current])
        # Check edges from others to current (since it's an undirected graph represented as a directed one)
        for node, edges in room_graph.items():
            if current in edges:
                neighbors.append(node)
                
        for neighbor in neighbors:
            if neighbor in room_ids and neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)
                
    return len(visited) == len(room_ids)


def drop_rooms(
    rooms: list[RoomDefinition], 
    room_graph: dict[str, dict[str, list[np.ndarray]]], 
    dropped_room_count: int = 2
) -> tuple[list[RoomDefinition], dict[str, dict[str, list[np.ndarray]]]]:
    """
    Randomly removes rooms from the layout while ensuring the remaining rooms stay connected.

    Args:
        rooms: The list of room definitions.
        room_graph: The current room graph.
        dropped_room_count: The number of rooms to attempt to remove.

    Returns:
        A tuple of (updated rooms list, updated room graph).
    """
    rooms_to_try = list(rooms)
    random.shuffle(rooms_to_try)
    
    dropped_so_far = 0
    for room in rooms_to_try:
        if dropped_so_far >= dropped_room_count:
            break
            
        # Try removing this room
        remaining_rooms = [r for r in rooms if r.room_id != room.room_id]
        remaining_ids = [r.room_id for r in remaining_rooms]
        
        # Check if graph remains connected
        if is_graph_connected(remaining_ids, room_graph):
            # Remove from rooms list
            rooms.remove(room)
            # Remove from graph
            if room.room_id in room_graph:
                del room_graph[room.room_id]
            for key in room_graph:
                if room.room_id in room_graph[key]:
                    del room_graph[key][room.room_id]
            dropped_so_far += 1
            
    return rooms, room_graph


def create_door_graph(
    room_graph: dict[str, dict[str, list[np.ndarray]]],
    extra_door_probability: float = 0.05
) -> dict[str, set[str]]:
    """
    Creates a graph representing door placements.
    Starts with a Minimum Spanning Tree (MST) from the room graph to ensure connectivity.
    Then randomly adds extra doors based on extra_door_probability.

    Args:
        room_graph: The room graph showing shared walls.
        extra_door_probability: Probability of adding an additional door between adjacent rooms.

    Returns:
        A dictionary mapping room IDs to a set of neighbor IDs they share a door with.
    """
    if not room_graph:
        return {}

    room_ids = list(room_graph.keys())
    start_node = random.choice(room_ids)

    visited = {start_node}
    door_graph = {room_id: set() for room_id in room_ids}

    # Edges are (weight, from_node, to_node)
    edges = []
    for neighbor in room_graph[start_node]:
        edges.append((random.random(), start_node, neighbor))

    heapq.heapify(edges)

    while edges and len(visited) < len(room_ids):
        weight, u, v = heapq.heappop(edges)

        if v in visited:
            continue

        visited.add(v)
        door_graph[u].add(v)
        door_graph[v].add(u)

        for next_neighbor in room_graph[v]:
            if next_neighbor not in visited:
                heapq.heappush(edges, (random.random(), v, next_neighbor))

    # Add random extra doors
    for u in room_graph:
        for v in room_graph[u]:
            if v not in door_graph[u]:
                if random.random() < extra_door_probability:
                    door_graph[u].add(v)
                    door_graph[v].add(u)

    return door_graph


def populate_doors(
    rooms: list[RoomDefinition],
    door_graph: dict[str, set[str]],
    room_graph: dict[str, dict[str, list[np.ndarray]]]
) -> None:
    """
    Populates the doors list in each RoomDefinition based on the door graph.
    Places a door at the center of the first shared edge segment between rooms.

    Args:
        rooms: The list of room definitions to update.
        door_graph: The graph of rooms that should have doors between them.
        room_graph: The graph containing shared wall segments.
    """
    room_map = {room.room_id: room for room in rooms}
    processed_pairs = set()

    for room_u_id, neighbors in door_graph.items():
        for room_v_id in neighbors:
            pair = tuple(sorted((room_u_id, room_v_id)))
            if pair in processed_pairs:
                continue
            processed_pairs.add(pair)

            shared_segments = room_graph[room_u_id][room_v_id]
            if not shared_segments:
                continue

            # Pick the first segment and place a door in the middle
            segment = shared_segments[0]
            door_pos = Point(
                x=(segment[0] + segment[2]) / 2,
                y=(segment[1] + segment[3]) / 2
            )
            door = Door(
                door_id=str(uuid.uuid4()),
                position=door_pos,
                room_ids={room_u_id, room_v_id}
            )

            room_map[room_u_id].doors.append(door)
            room_map[room_v_id].doors.append(door)
