from collections.abc import Callable
from typing import Any, Generic, TypeVar

import houston_server_gateways
from houston_server_gateways.utils import get_houston_data_root

T = TypeVar("T")


class PersistenceManager(Generic[T]):
    def __init__(self, data_key: str, dict_to_model: Callable[[dict[str, Any]], T]):
        self.key = data_key
        self.dict_to_model = dict_to_model

        houston_server_gateways.sqlite.create_table(self.key)
        houston_server_gateways.files.initialise()

    def save_upload(self, item_id, upload):
        return houston_server_gateways.files.save_upload(self.key, upload, item_id)

    def save_model_dir(self, item_id, upload):
        return houston_server_gateways.files.save_model_dir(self.key, upload, item_id)

    def save_model_file(self, item_id, upload, relative_path):
        return houston_server_gateways.files.save_model_file(
            self.key, upload, item_id, relative_path
        )

    def read_file(self, path: str) -> bytes | None:
        full_path = self.get_file_path(path)
        return houston_server_gateways.files.read_file(str(full_path))

    def zip_directory(self, path: str) -> bytes | None:
        full_path = self.get_file_path(path)
        return houston_server_gateways.files.zip_directory(str(full_path))

    def list_files(self, path: str) -> list[str]:
        full_path = self.get_file_path(path)
        return houston_server_gateways.files.list_files(str(full_path))

    def get_directory_size(self, path: str) -> int:
        full_path = self.get_file_path(path)
        return houston_server_gateways.files.get_directory_size(str(full_path))

    def move_file_to_dir(self, file_path: str, dir_path: str, new_name: str):
        full_file_path = self.get_file_path(file_path)
        full_dir_path = self.get_file_path(dir_path)
        houston_server_gateways.files.move_file_to_dir(
            str(full_file_path), str(full_dir_path), new_name
        )

    def get_file_path(self, path):
        return (get_houston_data_root() / self.key / path).absolute()

    def save_model(self, item_id, item: T) -> T:
        return houston_server_gateways.sqlite.put(
            self.key,
            item_id,
            item,
        )

    def get_item(self, item_id: str) -> T | None:
        return houston_server_gateways.sqlite.get(self.key, item_id, self.dict_to_model)

    def query_items(
        self,
        page: int,
        tags: list[str] | None = None,
        filter_dict: dict[str, Any] | None = None,
    ) -> tuple[list[T], int]:
        print(tags)
        return houston_server_gateways.sqlite.query(
            self.key, page, self.dict_to_model, tags=tags, filter_dict=filter_dict
        )

    def delete_item(self, item_id: str):
        item = self.get_item(item_id)
        if item:
            if hasattr(item, "path") and item.path:
                houston_server_gateways.files.delete_from_store(
                    self.get_file_path(item.path)
                )
            houston_server_gateways.sqlite.delete(self.key, item_id)
