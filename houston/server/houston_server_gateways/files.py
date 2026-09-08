import os
import shutil

from houston_server_gateways.utils import get_houston_data_root

REPLAY_STORE_PATH = get_houston_data_root()


def initialise():
    if not os.path.exists(REPLAY_STORE_PATH):
        os.makedirs(REPLAY_STORE_PATH)


def delete_from_store(path: str):
    """
    Deletes a file from the replay store.
    """
    if os.path.exists(path):
        os.remove(path)


def save_upload(directory_name, upload, model_id):
    file_type = upload.filename.split(".")[-1]
    file_name = model_id + "." + file_type
    file_path = REPLAY_STORE_PATH / directory_name / file_name

    os.makedirs(REPLAY_STORE_PATH / directory_name, exist_ok=True)

    with open(file_path, "wb+") as destination:
        destination.write(upload.file.read())

    return file_name
