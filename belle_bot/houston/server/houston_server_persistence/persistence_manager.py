import houston_server_gateways


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

    def query_items(self, page: int) -> tuple[list[T], int]:
        return houston_server_gateways.sqlite.query(
            self.key,
            page,
            self.dict_to_model
        )

    # def delete_replay(self, replay_id: str):
    #     replay = get_replay(replay_id)
    #     if replay:
    #         houston_server_gateways.files.delete_from_store(replay.path)
    #         houston_server_gateways.sqlite.delete(TABLE_NAME, replay_id)
    #
    # def save_upload(self, replay_id: str, upload: UploadFile) -> str:
    #     return houston_server_gateways.files.save_upload(
    #         TABLE_NAME,
    #         upload,
    #         replay_id
    #     )
