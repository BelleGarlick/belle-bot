import json
from collections import deque

from belle_bot.infra.fabric import FabricClient
import os
import queue
import threading
import time
import uuid
from pathlib import Path

CHUNK_TIME = 600  # 600s (10min)
LOG_ROOT_PATH = os.environ.get("REPLAYS_PATH", "replays")

# In-memory cache for the current chunk
_current_chunk_info = {
    "start_time": 0,
    "path": None
}

# Queue to offload SQLite writes away from the WebSocket thread
log_queue = queue.Queue(maxsize=250)


def _get_chunk_path(timestamp: float) -> Path | None:
    if not LOG_ROOT_PATH:
        return None

    global _current_chunk_info

    if (_current_chunk_info["path"] is None or
            timestamp - _current_chunk_info["start_time"] >= CHUNK_TIME):
        log_dir = Path(LOG_ROOT_PATH)
        log_dir.mkdir(parents=True, exist_ok=True)

        new_uuid = str(uuid.uuid4())
        print(f"Starting replay log: {new_uuid}")
        _current_chunk_info["start_time"] = timestamp
        _current_chunk_info["path"] = log_dir / f"{new_uuid}.txt"

    return _current_chunk_info["path"]


def _sqlite_writer_worker():
    """Background worker that continuously drains the log queue into SQLite."""
    while True:
        try:
            service_name, data, timestamp = log_queue.get(block=True)
            path = _get_chunk_path(timestamp)

            if path:
                with open(path, "a+") as f:
                    f.write(",".join([service_name, str(timestamp), json.dumps(data)]) + "\n")

        except Exception as e:
            print(f"Error in logging worker: {e}")
        finally:
            log_queue.task_done()


def cleanup_worker():
    """Background thread to handle 12-hour file cleanup periodically."""
    while True:
        if LOG_ROOT_PATH is not None and os.path.exists(LOG_ROOT_PATH):
            files = os.listdir(LOG_ROOT_PATH)
            now = time.time()
            for file in files:
                try:
                    file_path = os.path.join(LOG_ROOT_PATH, file)
                    if os.path.getmtime(file_path) < now - 3600 * 12:
                        print(f"Deleting old replay log: {file}")
                        os.remove(file_path)
                except Exception:
                    pass

        time.sleep(60)

rate = deque(maxlen=100)

def capture(x):
    """
    WebSocket callback: EXTREMELY FAST (<0.01ms).
    Directly puts data into memory queue without waiting for disk I/O.
    """
    now = time.time()
    service_name = x.get("service_name", "missing_service_name")

    rate.append(now)
    fps = 1 / ((rate[-1] - rate[0]) / len(rate))
    print("\rCPS: ", fps, end="")

    try:
        log_queue.put_nowait((service_name, x, now))
    except queue.Full:
        # Drop frame if storage can't keep up, keeping WebSocket loop alive
        pass


CLIENT = FabricClient()

if __name__ == "__main__":
    # Start background thread for writing to DB
    threading.Thread(target=_sqlite_writer_worker, daemon=True).start()

    # Start background thread for cleaning up old files
    threading.Thread(target=cleanup_worker, daemon=True).start()

    # Listen to WebSocket on main thread
    CLIENT.listen("*", capture)

    while True:
        pass
