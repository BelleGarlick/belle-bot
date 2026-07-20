import json
import os
import sqlite3
import time
import uuid
from pathlib import Path


CHUNK_TIME = 600  # 600s (10min)


# In-memory cache for the current chunk
_current_chunk_info = {
    "start_time": 0,
    "path": None
}


def _get_chunk_path(timestamp: float) -> Path | None:
    log_root_path = os.environ.get("REPLAYS_PATH")
    if not log_root_path:
        return None

    global _current_chunk_info

    # If we don't have a chunk or the current chunk has expired (10 mins)
    if (_current_chunk_info["path"] is None or 
        timestamp - _current_chunk_info["start_time"] >= 600):

        log_dir = Path(log_root_path)
        log_dir.mkdir(parents=True, exist_ok=True)

        new_uuid = str(uuid.uuid4())
        _current_chunk_info["start_time"] = timestamp
        _current_chunk_info["path"] = Path(log_dir / f"{new_uuid}.db")
    
    return _current_chunk_info["path"]


def initialise(path: Path):
    if path.exists():
        return

    # Crete the table since it doesn't exist
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS service_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            value BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log(service_name: str, data: dict[str, str]):
    now = time.time()
    # Force use time.time() explicitly in log to pick up mock if present
    path = _get_chunk_path(now)
    if not path:
        return

    initialise(path)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO service_logs (service_name, timestamp, value) VALUES (?, ?, ?)",
        (service_name, str(now), json.dumps(data))
    )

    conn.commit()
    conn.close()

