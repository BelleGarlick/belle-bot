import numpy as np

from belle_bot.mapping.world_generation.inside_generation.rooms_layout_generation.models import RoomDefinition, Point, \
    Window


def get_window_candidates(
    rooms: list[RoomDefinition], 
    room_graph: dict[str, dict[str, list[np.ndarray]]]
) -> dict[str, dict[str, list[np.ndarray]]]:
    """
    Finds segments of room edges that are external (not shared with other rooms).
    These are candidates for windows.

    Args:
        rooms: The list of room definitions.
        room_graph: The room graph containing shared wall information.

    Returns:
        A dictionary mapping room IDs to their external wall segments categorized by side (top, bottom, left, right).
    """
    candidates = {}

    for room in rooms:
        # Each edge is [x1, y1, x2, y2]
        edges = {
            'top': [room.top_edge],
            'bottom': [room.bottom_edge],
            'left': [room.left_edge],
            'right': [room.right_edge]
        }

        # Shared edges with neighbors
        if room.room_id in room_graph:
            for neighbor_id in room_graph[room.room_id]:
                shared_edges = room_graph[room.room_id][neighbor_id]
                for shared in shared_edges:
                    # Subtract shared segment from our edges
                    for edge_type in edges:
                        new_segments = []
                        for segment in edges[edge_type]:
                            res = subtract_segment(segment, shared)
                            new_segments.extend(res)
                        edges[edge_type] = new_segments
        
        candidates[room.room_id] = edges
    
    return candidates


def populate_windows(
    rooms: list[RoomDefinition],
    window_candidates: dict[str, dict[str, list[np.ndarray]]]
) -> None:
    """
    Populates the windows list in each RoomDefinition based on window candidates.
    Places a window at the center of each external wall segment.

    Args:
        rooms: The list of room definitions to update.
        window_candidates: External wall segments for each room.
    """
    room_map = {room.room_id: room for room in rooms}

    for room_id, edges in window_candidates.items():
        room = room_map[room_id]
        for edge_type in edges:
            segments = edges[edge_type]
            for segment in segments:
                # Place a window in the middle of each external segment
                window_pos = Point(
                    x=(segment[0] + segment[2]) / 2,
                    y=(segment[1] + segment[3]) / 2
                )
                room.windows.append(Window(position=window_pos))


def subtract_segment(full_edge: np.ndarray, shared_segment: np.ndarray) -> list[np.ndarray]:
    """
    Subtracts shared_segment from full_edge. Both are [x1, y1, x2, y2].

    Args:
        full_edge: The original wall segment.
        shared_segment: The segment to remove (shared with another room).

    Returns:
        A list of remaining wall segments.
    """
    x1, y1, x2, y2 = full_edge
    sx1, sy1, sx2, sy2 = shared_segment

    # Check if they are collinear
    # Horizontal
    if y1 == y2 == sy1 == sy2:
        # Sort x coordinates
        min_f, max_f = min(x1, x2), max(x1, x2)
        min_s, max_s = min(sx1, sx2), max(sx1, sx2)

        # No overlap
        if max_s <= min_f or min_s >= max_f:
            return [full_edge]

        segments = []
        if min_s > min_f:
            segments.append(np.array([min_f, y1, min_s, y1]))
        if max_s < max_f:
            segments.append(np.array([max_s, y1, max_f, y1]))
        return segments

    # Vertical
    if x1 == x2 == sx1 == sx2:
        # Sort y coordinates
        min_f, max_f = min(y1, y2), max(y1, y2)
        min_s, max_s = min(sy1, sy2), max(sy1, sy2)

        # No overlap
        if max_s <= min_f or min_s >= max_f:
            return [full_edge]

        segments = []
        if min_s > min_f:
            segments.append(np.array([x1, min_f, x1, min_s]))
        if max_s < max_f:
            segments.append(np.array([x1, max_s, x1, max_f]))
        return segments

    return [full_edge]
