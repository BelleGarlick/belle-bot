import asyncio
import base64
import orjson  # faster than json
import httpx
import numpy as np
import requests
import websockets

from .utils import FABRIC_PORT


def process_data(data):
    output = {}

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            value = base64.b64encode(value.tobytes()).decode("utf-8")

        if isinstance(value, int) or isinstance(value, float):
            value = str(value)

        if value is None:
            value = 'null'

        output[key] = value

    return output


class FabricClient:
    def __init__(self, host="localhost"):
        self.host = host
        self.port = FABRIC_PORT
        self.__listeners = {}

    async def listen_async(self, stream: str, callback):
        ws_url = f"ws://{self.host}:{self.port}/listen/{stream}"

        async for ws in websockets.connect(ws_url, ping_interval=None):
            try:
                async for message in ws:
                    callback(orjson.loads(message))
            except websockets.ConnectionClosed:
                await asyncio.sleep(0.1)

    def publish(self, stream: str, data: dict):
        data = process_data(data)

        url = f"http://{self.host}:{self.port}/publish/{stream}"
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to publish to {stream}: {e}")
            return None

    async def publish_async(self, service_name, data):
        data = process_data(data)

        # Using httpx for async publishing
        url = f"http://{self.host}:{self.port}/publish/{service_name}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json=data)
                response.raise_for_status()
            except Exception as e:
                print(f"Failed to publish to {service_name}: {e}")
