from houston.server import houston_server_gateways
from houston.server.houston_server_models.replay import Replay


# todo handle the file objects


TABLE_NAME = "REPLAYS"


def initialise():
    houston_server_gateways.sqlite.create_table(TABLE_NAME)
    houston_server_gateways.files.initialise(TABLE_NAME)


def save_replay(replay: Replay) -> Replay:
    return houston_server_gateways.sqlite.put(
        TABLE_NAME,
        replay.replay_id,
        replay,
    )


def get_replay(replay_id: str) -> Replay | None:
    return houston_server_gateways.sqlite.get(
        TABLE_NAME,
        replay_id,
        lambda x: Replay(**x)
    )


def query_replays(page: int) -> tuple[list[Replay], int]:
    return houston_server_gateways.sqlite.query(
        TABLE_NAME,
        page,
        lambda x: Replay(**x)
    )


def delete_replay(replay_id: str):
    replay = get_replay(replay_id)
    if replay:
        houston_server_gateways.files.delete_from_store(replay.path)
        houston_server_gateways.sqlite.delete(TABLE_NAME, replay_id)


def move_replay_file(source_path: str, replay_file_id: str):
    # todo figure this out more. maybe handled in the core
    houston_server_gateways.files.move_to_store(source_path, TABLE_NAME + "/" + replay_file_id + "." + file_type)
