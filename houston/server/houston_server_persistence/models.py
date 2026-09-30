from datetime import datetime

from pydantic import BaseModel


class Model(BaseModel):
    model_id: str

    # the model name amongst multiple models should remain the same such that we can
    # query the model name and it return all versions
    name: str

    # Path to the model
    path: str

    tags: list[str]

    version: str

    # this should include things like how to obtain the model
    # how it's used, yada yada yada
    description: str | None = None

    upload_time: datetime

    # The file size or total size of directory
    size: int

    # Is it a directory
    is_dir: bool = False
