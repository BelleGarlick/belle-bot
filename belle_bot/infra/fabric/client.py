import asyncio
import base64
import socket

import orjson  # faster than json
import threading
import traceback

import httpx
import numpy as np
import websocket
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

        # open connection with disabled ping timeouts and no delay
        async for ws in websockets.connect(ws_url, ping_interval=None):
            try:
                async for message in ws:
                    callback(orjson.loads(message))
            except websockets.ConnectionClosed:
                # reconnect
                await asyncio.sleep(0.1)

    def listen(self, stream: str, callback):
        if stream in self.__listeners:
            return

        def on_open(ws):
            ws.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        def on_message(ws, message):
            callback(orjson.loads(message))

        def on_error(ws, error):
            print(f"WebSocket error: {error!r}")

        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed for stream {stream}")

        def run_ws():
            ws_url = f"ws://{self.host}:{self.port}/listen/{stream}"
            while True:
                try:
                    ws = websocket.WebSocketApp(
                        ws_url,
                        on_message=on_message,
                        on_open=on_open,
                        on_error=on_error,
                        on_close=on_close,
                    )

                    self.__listeners[stream] = ws

                    ws.run_forever(skip_utf8_validation=True)

                except Exception:
                    traceback.print_exc()

                print(f"Stream ended reconnecting to {stream}")


        wst = threading.Thread(target=run_ws)
        wst.daemon = True
        wst.start()

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
