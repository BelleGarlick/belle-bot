import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

# Default directory for logs
DEFAULT_LOG_DIR = "replays"

# In-memory cache for the current chunk
_current_chunk_info = {
    "start_time": 0,
    "path": None
}


def _get_chunk_path(timestamp: float = None):
    # Check if logging is enabled
    enabled = os.environ.get("SAVE_REPLAY_ENABLED", "false").lower() == "true"
    if not enabled:
        return None
    
    if timestamp is None:
        timestamp = time.time()
    
    global _current_chunk_info
    
    log_dir = Path(DEFAULT_LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    # If we don't have a chunk or the current chunk has expired (10 mins)
    if (_current_chunk_info["path"] is None or 
        timestamp - _current_chunk_info["start_time"] >= 600 or
        _current_chunk_info["path"].parent.resolve() != log_dir.resolve()):
        
        # Try to find the most recent chunk in the directory to see if we can continue it
        # (in case of process restart within the same 10-min window)
        # However, the user asked for UUIDs, and typically chunks are sequential.
        # To keep it simple and follow "each file use a uuidv4", we'll just create a new one
        # if the in-memory one is missing or expired.
        
        new_uuid = str(uuid.uuid4())
        _current_chunk_info["start_time"] = (timestamp // 600) * 600
        _current_chunk_info["path"] = log_dir / f"{new_uuid}.db"
    
    return _current_chunk_info["path"]


def initialise(timestamp: float = None):
    path = _get_chunk_path(timestamp)
    if not path:
        return

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

    # Ensure table exists for this chunk
    initialise(now)

    conn = sqlite3.connect(path)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO service_logs (service_name, timestamp, value) VALUES (?, ?, ?)",
        (service_name, str(now), json.dumps(data))
    )

    conn.commit()
    conn.close()

