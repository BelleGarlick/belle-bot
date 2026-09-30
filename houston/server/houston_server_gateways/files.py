import os
import shutil

from houston_server_gateways.utils import get_houston_data_root

REPLAY_STORE_PATH = get_houston_data_root()


def initialise():
    if not os.path.exists(REPLAY_STORE_PATH):
        os.makedirs(REPLAY_STORE_PATH)


def delete_from_store(path: str):
    """
    Deletes a file or directory from the replay store.
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def read_file(path: str) -> bytes | None:
    """
    Reads a file from the store.
    """
    if not os.path.exists(path) or not os.path.isfile(path):
        return None

    # Security check: ensure path is within REPLAY_STORE_PATH
    if not os.path.abspath(path).startswith(os.path.abspath(REPLAY_STORE_PATH)):
        return None

    with open(path, "rb") as f:
        return f.read()


def zip_directory(path: str) -> bytes | None:
    """
    Zips a directory and returns the bytes.
    """
    if not os.path.exists(path) or not os.path.isdir(path):
        return None

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        zip_base = os.path.join(temp_dir, "archive")
        shutil.make_archive(zip_base, "zip", path)
        with open(f"{zip_base}.zip", "rb") as f:
            return f.read()


def list_files(path: str) -> list[str]:
    """
    Lists all files in a directory recursively, returning relative paths.
    """
    if not os.path.exists(path):
        return []
    if not os.path.isdir(path):
        return [os.path.basename(path)]

    files = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            rel_path = os.path.relpath(os.path.join(root, filename), path)
            files.append(rel_path)
    return files


def get_directory_size(path: str) -> int:
    """
    Calculates the total size of a directory.
    """
    total_size = 0
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            total_size += os.path.getsize(os.path.join(root, filename))
    return total_size


def move_file_to_dir(file_path: str, dir_path: str, new_name: str):
    """
    Moves a file into a directory.
    """
    os.makedirs(dir_path, exist_ok=True)
    shutil.move(file_path, os.path.join(dir_path, new_name))


def save_upload(directory_name, upload, model_id):
    file_type = upload.filename.split(".")[-1]
    file_name = model_id + "." + file_type
    file_path = REPLAY_STORE_PATH / directory_name / file_name

    os.makedirs(file_path.parent, exist_ok=True)

    with open(file_path, "wb+") as destination:
        shutil.copyfileobj(upload.file, destination)

    return file_name


def save_model_file(directory_name, upload, model_id, relative_path):
    """
    Saves an individual file to a model directory.
    """
    dir_path = REPLAY_STORE_PATH / directory_name / model_id
    file_path = dir_path / relative_path
    
    # Security check: ensure file_path is within dir_path
    if not str(file_path.resolve()).startswith(str(dir_path.resolve())):
        raise ValueError("Invalid relative path")

    os.makedirs(file_path.parent, exist_ok=True)

    with open(file_path, "wb+") as destination:
        shutil.copyfileobj(upload.file, destination)

    return str(relative_path)


def save_model_dir(directory_name, upload, model_id):
    """
    Saves a model directory. If upload is a zip file, it extracts it.
    """
    file_type = upload.filename.split(".")[-1]
    if file_type.lower() == "zip":
        import zipfile
        import io
        
        dir_path = REPLAY_STORE_PATH / directory_name / model_id
        os.makedirs(dir_path, exist_ok=True)
        
        with zipfile.ZipFile(io.BytesIO(upload.file.read())) as zip_ref:
            zip_ref.extractall(dir_path)
        
        return str(model_id)
    else:
        # Fallback to single file save if not a zip
        return save_upload(directory_name, upload, model_id)
