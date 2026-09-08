from urllib.parse import urlencode

from houston.client.py.config import HoustonConfig
from houston.client.py.utils import get


def query_replays(config: HoustonConfig, page: int, tags: list[str] | None = None) -> list[dict]:
    params = {}
    if page is not None:
        params["page"] = page
    if tags:
        params["tags"] = tags

    query_string = urlencode(params, doseq=True)
    url = f"/api/replays?{query_string}" if query_string else "/replays"

    return get(config, url)


def get_replay_file(config: HoustonConfig, replay_id: str):
    return get(config, f"/api/replays/{replay_id}", json=False)
