import requests


HOUSTON_URL = "http://localhost:8080"


def get(url, json=True):
    response = requests.get(HOUSTON_URL + url)
    if json:
        return response.json()
    return response.text
