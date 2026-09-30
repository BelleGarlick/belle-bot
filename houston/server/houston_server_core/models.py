import datetime
import os
import uuid

import pytz
from fastapi import UploadFile
from houston_server_persistence import PersistenceManager
from houston_server_persistence.models import Model


def get_model_persistence() -> PersistenceManager[Model]:
    return PersistenceManager[Model]("models", lambda data: Model(**data))


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

    is_zip = upload.filename.lower().endswith(".zip")
    if is_zip:
        path = get_model_persistence().save_model_dir(model_id, upload)
    else:
        path = get_model_persistence().save_upload(model_id, upload)

    return get_model_persistence().save_model(
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
            is_dir=is_zip,
        ),
    )


def get_model_object(model: Model) -> bytes | None:
    if model.is_dir:
        return get_model_persistence().zip_directory(model.path)
    return get_model_persistence().read_file(model.path)


def get_model(model_id: str) -> Model | None:
    return get_model_persistence().get_item(model_id)


def add_model_file(
    model_id: str,
    relative_path: str,
    upload: UploadFile,
) -> Model:
    model = get_model(model_id)
    if not model:
        raise ValueError("Model not found")

    if not model.is_dir:
        # If it was a single file, we need to move it to a directory
        # model.path was like {uuid}.{ext} and we want to move it to a dir named {uuid}
        get_model_persistence().move_file_to_dir(model.path, model_id, model.path)
        model.path = model_id
        model.is_dir = True

    # Save the new file
    get_model_persistence().save_model_file(model_id, upload, relative_path)

    # Recalculate size
    model.size = get_model_persistence().get_directory_size(model.path)

    return get_model_persistence().save_model(model_id, model)


def query_models(
    page: int, tags: list[str] | None = None, name: str | None = None
) -> tuple[list[Model], int]:
    filter_dict = {}
    if name:
        filter_dict["name"] = name
    return get_model_persistence().query_items(page, tags=tags, filter_dict=filter_dict)


def list_model_files(model: Model) -> list[str]:
    return get_model_persistence().list_files(model.path)


def get_model_file(model: Model, relative_path: str | None = None) -> bytes | None:
    if not model.is_dir:
        return get_model_persistence().read_file(model.path)

    if not relative_path:
        return None

    return get_model_persistence().read_file(os.path.join(model.path, relative_path))
