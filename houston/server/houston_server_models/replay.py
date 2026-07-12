import datetime
from typing import Optional

from pydantic import BaseModel


class Replay(BaseModel):

    replay_id: str

    firmware_version: str

    hardware_version: str

    filename: str

    path: str

    description: Optional[str] = None

    start_time: datetime.datetime

    end_time: datetime.datetime

    permanent: bool = False

    tags: list[str] = []
