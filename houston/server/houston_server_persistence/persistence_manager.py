import houston_server_gateways
from houston_server_gateways.utils import get_houston_data_root


class PersistenceManager[T]:

    def __init__(self, data_key: str, dict_to_model):
        self.key = data_key
        self.dict_to_model = dict_to_model

        houston_server_gateways.sqlite.create_table(self.key)
        houston_server_gateways.files.initialise()

    def save_upload(self, item_id, upload):
        return houston_server_gateways.files.save_upload(
            self.key,
            upload,
            item_id
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
        return houston_server_gateways.sqlite.get(
            self.key,
            item_id,
            self.dict_to_model
        )

    def query_items(self, page: int, tags: list[str] | None = None) -> tuple[list[T], int]:
        print(tags)
        return houston_server_gateways.sqlite.query(
            self.key,
            page,
            self.dict_to_model,
            tags=tags
        )

    def delete_item(self, item_id: str):
        item = self.get_item(item_id)
        if item:
            if hasattr(item, 'path') and item.path:
                houston_server_gateways.files.delete_from_store(self.get_file_path(item.path))
            houston_server_gateways.sqlite.delete(self.key, item_id)
