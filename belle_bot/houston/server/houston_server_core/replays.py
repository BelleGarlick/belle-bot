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


def upload_replay(
    filename: str,
    firmware_version: str,
    physical_device: str,
    permanent: bool,
    tags: list[str],
    upload: UploadFile,
    description: str | None = None,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> Replay:
    replay_id = str(uuid.uuid4())

    path = replays_persistence.save_upload(replay_id, upload)

    return replays_persistence.save_model(
        replay_id,
        Replay(
            filename=filename,
            path=path,
            replay_id=replay_id,
            firmware_version=firmware_version,
            hardware_version=firmware_version,
            description=description,
            start_time=start_time,
            end_time=end_time,
            permanent=permanent,
            tags=tags,
            physical_device=physical_device,
            upload_time=datetime.datetime.now(tz=pytz.utc),
        )
    )


def get_replay(replay_id: str) -> Replay | None:
    replays_persistence.get_item(replay_id)


def query_replays(page: int) -> tuple[list[Replay], int]:
    return replays_persistence.query_items(page)
