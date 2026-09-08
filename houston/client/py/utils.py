import requests

from houston.client.py.config import HoustonConfig


def get(config: HoustonConfig, url, json=True):
    response = requests.get(config.endpoint + url)
    if json:
        return response.json()
    return response.text
