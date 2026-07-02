import json
import threading

import httpx
import websocket
import requests
from .utils import FABRIC_PORT


class FabricClient:
    def __init__(self, host="localhost"):
        self.host = host
        self.port = FABRIC_PORT
        self.__listeners = {}

    def listen(self, stream: str, callback):
        if stream in self.__listeners:
            return

        def on_message(ws, message):
            data = json.loads(message)
            callback(data.get("data", {}))

        def on_error(ws, error):
            print(f"WebSocket error for stream {stream}: {error}")

        def on_close(ws, close_status_code, close_msg):
            print(f"WebSocket closed for stream {stream}")

        def run_ws():
            ws_url = f"ws://{self.host}:{self.port}/listen/{stream}"
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            self.__listeners[stream] = ws
            ws.run_forever()

        wst = threading.Thread(target=run_ws)
        wst.daemon = True
        wst.start()

    def publish(self, stream: str, data: dict):
        url = f"http://{self.host}:{self.port}/publish/{stream}"
        try:
            response = requests.post(url, json={"data": data})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to publish to {stream}: {e}")
            return None

    async def publish_async(self, service_name, data):
        # Using httpx for async publishing
        url = f"http://{self.host}:{self.port}/publish/{service_name}"
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(url, json={"data": data})
                response.raise_for_status()
            except Exception as e:
                print(f"Failed to publish to {service_name}: {e}")
