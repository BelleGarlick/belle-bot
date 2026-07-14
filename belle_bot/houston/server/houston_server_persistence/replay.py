import datetime

from pydantic import BaseModel


class Replay(BaseModel):

    replay_id: str

    firmware_version: str

    hardware_version: str

    physical_device: str

    filename: str

    path: str

    description: str | None = None

    start_time: datetime.datetime | None = None

    end_time: datetime.datetime | None = None

    upload_time: datetime.datetime

    permanent: bool = False

    tags: list[str] = []
