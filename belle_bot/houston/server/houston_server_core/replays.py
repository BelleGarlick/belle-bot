import datetime
import uuid

import pytz
from fastapi import UploadFile

from houston_server_persistence.replay import Replay
from houston_server_persistence import PersistenceManager


replays_persistence = PersistenceManager[Replay](
    "replays",
    lambda data: Replay(**data)
)

from os import SEEK_END, SEEK_CUR


def readlast(f):
    try:
        f.seek(-2, SEEK_END)  # Jump to the second last byte.
        while f.read(1) != b"\n":  # Until newline is found ...
            f.seek(-2, SEEK_CUR)  # ... jump back, over the read byte plus one.
    except OSError:  # Reached begginning of File
        f.seek(0)  # Set cursor to beginning of file as well.
    return f.read()  # Read all data from this point on.


def upload_replay(
    upload: UploadFile,
    tags: list[str],
    platform: str | None = None,
    filename: str | None = None,
    permanent: bool = False,
    description: str | None = None
) -> Replay:
    replay_id = str(uuid.uuid4())

    path = replays_persistence.save_upload(replay_id, upload)

    with open(replays_persistence.get_file_path(path), "rb") as f:
        first = f.readline().decode("utf-8")
        last = readlast(f).decode("utf-8")

    # Parse the time stamps
    start_time = datetime.datetime.fromtimestamp(float(first.split(",")[1]))
    end_time = datetime.datetime.fromtimestamp(float(last.split(",")[1]))

    return replays_persistence.save_model(
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
    return replays_persistence.get_item(replay_id)


def get_replay_object(replay: Replay) -> str | None:
    file_path = replays_persistence.get_file_path(replay.path)
    if file_path.exists():
        return str(file_path)
    return None


def update_replay(replay_id: str, replay: Replay) -> Replay | None:
    existing = get_replay(replay_id)
    if not existing:
        return None
    return replays_persistence.save_model(replay_id, replay)


def query_replays(page: int, tags: list[str] | None = None) -> tuple[list[Replay], int]:
    return replays_persistence.query_items(page, tags=tags)
