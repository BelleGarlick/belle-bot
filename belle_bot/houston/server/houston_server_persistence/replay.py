import datetime

from pydantic import BaseModel


class Replay(BaseModel):

    replay_id: str

    platform: str | None = None

    filename: str | None = None

    path: str

    description: str | None = None

    start_time: datetime.datetime | None = None

    end_time: datetime.datetime | None = None

    upload_time: datetime.datetime

    permanent: bool = False

    tags: list[str] = []
