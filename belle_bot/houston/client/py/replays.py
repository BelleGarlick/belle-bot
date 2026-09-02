from urllib.parse import urlencode

from belle_bot.houston.client.py.utils import get


def query_replays(page: int, tags: list[str] | None = None) -> list[dict]:
    params = {}
    if page is not None:
        params["page"] = page
    if tags:
        params["tags"] = tags

    query_string = urlencode(params, doseq=True)
    url = f"/replays?{query_string}" if query_string else "/replays"

    return get(url)


def get_replay_file(replay_id: str):
    return get(f"/replays/{replay_id}", json=False)
