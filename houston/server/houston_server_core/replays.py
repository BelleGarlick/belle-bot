from typing import Any
import houston_server_persistence as persistence

persistence.replays.initialise()

def upload_replay(
    filename: str, 
    content: bytes, 
    description: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    permanent: bool = False,
    tags: list[str] = None
) -> int:
    return persistence.replays.save_replay(
        filename=filename,
        content=content,
        description=description,
        start_time=start_time,
        end_time=end_time,
        permanent=permanent,
        tags=tags or []
    )

def get_replay(replay_id: int) -> dict[str, Any] | None:
    return persistence.replays.get_replay(replay_id)

def list_replays(
    tag: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
    permanent: bool | None = None
) -> list[dict[str, Any]]:
    return persistence.replays.query_replays(
        tag=tag,
        start_time=start_time,
        end_time=end_time,
        permanent=permanent
    )
