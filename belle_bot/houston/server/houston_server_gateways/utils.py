import os
from pathlib import Path


def get_houston_data_root():
    return Path(os.environ["HOUSTON_PATH"])
