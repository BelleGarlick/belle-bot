import datetime
import uuid

import pytz
from fastapi import UploadFile

from houston_server_persistence.replay import Replay
from houston_server_persistence import PersistenceManager


def get_replay_persistence() -> PersistenceManager[Replay]:
    return PersistenceManager[Replay](
        "replays",
        lambda data: Replay(**data)
    )

from os import SEEK_END, SEEK_CUR


def readlast(f):
    f.seek(0, SEEK_END)
    file_size = f.tell()
    if file_size == 0:
        return b""
    
    buffer_size = 1024
    offset = 0
    
    while True:
        offset += buffer_size
        if offset >= file_size:
            f.seek(0)
            return f.read()
        
        f.seek(-offset, SEEK_END)
        buffer = f.read(buffer_size)
        
        newline_pos = buffer.rfind(b"\n")
        if newline_pos != -1:
            # Found a newline. If it's the very last byte of the file, we need to keep looking
            # unless it's the only newline.
            if offset == buffer_size and newline_pos == buffer_size - 1:
                # Last byte is newline, check if there's another one in this buffer
                second_last_newline = buffer[:newline_pos].rfind(b"\n")
                if second_last_newline != -1:
                    f.seek(-offset + second_last_newline + 1, SEEK_END)
                    return f.read()
                else:
                    # Only the trailing newline found so far, continue searching in next block
                    continue
            
            f.seek(-offset + newline_pos + 1, SEEK_END)
            return f.read()


def upload_replay(
    upload: UploadFile,
    tags: list[str],
    platform: str | None = None,
    filename: str | None = None,
    permanent: bool = False,
    description: str | None = None
) -> Replay:
    replay_id = str(uuid.uuid4())

    path = get_replay_persistence().save_upload(replay_id, upload)

    with open(get_replay_persistence().get_file_path(path), "rb") as f:
        first = f.readline().decode("utf-8")
        last = readlast(f).decode("utf-8")

    # Parse the time stamps
    start_time = datetime.datetime.fromtimestamp(float(first.split(",")[1]))
    end_time = datetime.datetime.fromtimestamp(float(last.split(",")[1]))

    return get_replay_persistence().save_model(
        replay_id,
        Replay(
            filename=filename,
            path=path,
            replay_id=replay_id,
            platform=platform,
            description=description,
            start_time=start_time,
            end_time=end_time,
            permanent=permanent,
            tags=tags,
            upload_time=datetime.datetime.now(tz=pytz.utc),
        )
    )


def get_replay(replay_id: str) -> Replay | None:
    return get_replay_persistence().get_item(replay_id)


def get_replay_object(replay: Replay) -> str | None:
    file_path = get_replay_persistence().get_file_path(replay.path)
    if file_path.exists():
        return str(file_path)
    return None


def update_replay(replay_id: str, replay: Replay) -> Replay | None:
    existing = get_replay(replay_id)
    if not existing:
        return None
    return get_replay_persistence().save_model(replay_id, replay)


def query_replays(page: int, tags: list[str] | None = None) -> tuple[list[Replay], int]:
    return get_replay_persistence().query_items(page, tags=tags)


def delete_replay(replay_id: str):
    get_replay_persistence().delete_item(replay_id)
