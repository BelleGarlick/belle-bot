import datetime
import uuid

import pytz
from fastapi import UploadFile
from houston_server_persistence import PersistenceManager
from houston_server_persistence.dataset import Dataset


def get_dataset_persistence() -> PersistenceManager[Dataset]:
    """
    Returns the PersistenceManager instance for Dataset models.

    This manager handles the interaction between the Dataset model and the
    underlying storage (SQLite for metadata and filesystem for files).
    """
    return PersistenceManager[Dataset]("datasets", lambda data: Dataset(**data))


def upload_dataset(
    upload: UploadFile,
    tags: list[str],
    description: str | None = None,
    path_override: str | None = None,
) -> Dataset:
    """
    Uploads a new dataset and saves its metadata.

    Args:
        upload: The uploaded file object.
        tags: A list of tags to associate with the dataset.
        description: An optional description of the dataset.
        path_override: An optional custom path to save the file at. If not provided,
            a UUID will be used.

    Returns:
         The created Dataset model instance.
    """
    dataset_id = str(uuid.uuid4())

    # Use path_override if provided (e.g., vision/encoder/v1/train)
    # The file gateway might need to handle directories if path_override has slashes
    storage_path = path_override or dataset_id

    saved_path = get_dataset_persistence().save_upload(storage_path, upload)

    return get_dataset_persistence().save_model(
        dataset_id,
        Dataset(
            dataset_id=dataset_id,
            path=saved_path,
            description=description,
            tags=tags,
            upload_time=datetime.datetime.now(tz=pytz.utc),
        ),
    )


def get_dataset(dataset_id: str) -> Dataset | None:
    """
    Retrieves a dataset by its ID.

    Args:
        dataset_id: The unique identifier of the dataset.

    Returns:
        The Dataset instance if found, None otherwise.
    """
    return get_dataset_persistence().get_item(dataset_id)


def update_dataset(dataset_id: str, dataset: Dataset) -> Dataset | None:
    """
    Updates an existing dataset's metadata.

    Args:
        dataset_id: The unique identifier of the dataset to update.
        dataset: The updated Dataset model instance.

    Returns:
        The updated Dataset instance if successful, None if the dataset was not found.
    """
    existing = get_dataset(dataset_id)
    if not existing:
        return None
    return get_dataset_persistence().save_model(dataset_id, dataset)


def get_dataset_file_path(dataset: Dataset) -> str | None:
    """
    Returns the absolute path to the dataset file on disk.

    Args:
        dataset: The Dataset model instance.

    Returns:
        The absolute path as a string if the file exists, None otherwise.
    """
    file_path = get_dataset_persistence().get_file_path(dataset.path)
    if file_path.exists():
        return str(file_path)
    return None


def query_datasets(
    page: int, tags: list[str] | None = None
) -> tuple[list[Dataset], int]:
    """
    Queries datasets with optional pagination and tag filtering.

    Args:
        page: The page number to retrieve (0-indexed).
        tags: An optional list of tags to filter by.

    Returns:
        A tuple containing a list of Dataset instances and the total count of matches.
    """
    return get_dataset_persistence().query_items(page, tags=tags)


def delete_dataset(dataset_id: str):
    """
    Deletes a dataset and its associated file.

    Args:
        dataset_id: The unique identifier of the dataset to delete.
    """
    get_dataset_persistence().delete_item(dataset_id)
