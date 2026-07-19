import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from belle_bot.infra.fabric import utils

# todo enable a key

app = FastAPI()

# In-memory storage for active websocket connections: stream_name -> list of WebSocket
ACTIVE_CONNECTIONS: dict[str, list[WebSocket]] = {}

class Message(BaseModel):
    data: dict[str, str]  # data is map of str -> base64 encoded bytes

@app.websocket("/listen/{stream:path}")
async def websocket_endpoint(websocket: WebSocket, stream: str):
    await websocket.accept()
    if stream not in ACTIVE_CONNECTIONS:
        ACTIVE_CONNECTIONS[stream] = []
    ACTIVE_CONNECTIONS[stream].append(websocket)
    try:
        while True:
            # Keep the connection open and wait for messages (though we mostly push)
            await websocket.receive_text()
    except WebSocketDisconnect:
        ACTIVE_CONNECTIONS[stream].remove(websocket)
        if not ACTIVE_CONNECTIONS[stream]:
            del ACTIVE_CONNECTIONS[stream]

@app.post("/publish/{stream:path}")
async def publish(stream: str, message: Message):
    now = int(time.time())

    # Get all connections that match the key prefix
    connections = []
    for key in list(ACTIVE_CONNECTIONS):
        key_prefix = key.split("*")[0]
        if stream.startswith(key_prefix):
            connections += ACTIVE_CONNECTIONS[key]

    # Iterate through the matched connections to broadcast out the data
    disconnected = []
    for conn in connections:
        try:
            await conn.send_json({
                "stream": stream,
                "timestamp": now,
                "data": message.model_dump()
            })
        except Exception:
            disconnected.append(conn)

    # Clean up stale connections
    for conn in disconnected:
        ACTIVE_CONNECTIONS[stream].remove(conn)

    return {"status": "published", "listener_count": len(ACTIVE_CONNECTIONS.get(stream, []))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=utils.FABRIC_PORT)
