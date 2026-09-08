import datetime
import uuid

import pytz
from fastapi import UploadFile

from houston_server_persistence.models import Model
from houston_server_persistence import PersistenceManager

models_persistence = PersistenceManager[Model](
    "models",
    lambda data: Model(**data)
)


def upload_model(
    name: str,
    version: str,
    tags: list[str],
    description: str,
    upload: UploadFile,
) -> Model:
    if upload.size < 1:
        raise ValueError("Upload file issue. File is empty.")

    model_id = str(uuid.uuid4())

    path = models_persistence.save_upload(model_id, upload)

    return models_persistence.save_model(
        model_id,
        Model(
            model_id=model_id,
            path=path,
            name=name,
            tags=tags,
            version=version,
            description=description,
            upload_time=datetime.datetime.now(tz=pytz.utc),
            size=upload.size or -1,
        )
    )


def get_model(model_id: str) -> Model | None:
    return models_persistence.get_item(model_id)


def query_models(page: int) -> tuple[list[Model], int]:
    return models_persistence.query_items(page)
