import datetime

from pydantic import BaseModel


class Dataset(BaseModel):
    dataset_id: str
    path: str
    description: str | None = None
    upload_time: datetime.datetime
    tags: list[str] = []
