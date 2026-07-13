import os
import shutil


REPLAY_STORE_PATH = os.environ.get("REPLAY_STORE_PATH", "replays")


def initialise():
    if not os.path.exists(REPLAY_STORE_PATH):
        os.makedirs(REPLAY_STORE_PATH)


def move_to_store(source_path: str, filename: str) -> str:
    """
    Moves a file from source_path to the replay store.
    Returns the new path of the file.
    """
    initialise()
    destination_path = os.path.join(REPLAY_STORE_PATH, filename)
    shutil.move(source_path, destination_path)
    return destination_path


def delete_from_store(path: str):
    """
    Deletes a file from the replay store.
    """
    if os.path.exists(path):
        os.remove(path)


def save_upload(directory_name, upload, model_id):
    file_type = upload.filename.split(".")[-1]
    file_path = directory_name + "/" + model_id + "." + file_type

    with open(file_path, "wb+") as destination:
        destination.write(upload.file.read())

    return file_path
