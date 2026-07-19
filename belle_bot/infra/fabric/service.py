from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from belle_bot.infra.fabric import logs, utils

app = FastAPI()


# In-memory storage for active websocket connections: stream_name -> list of WebSocket
active_connections: dict[str, list[WebSocket]] = {}

class Message(BaseModel):
    data: dict[str, str]  # data is map of str -> base64 encoded bytes

@app.websocket("/listen/{stream:path}")
async def websocket_endpoint(websocket: WebSocket, stream: str):
    await websocket.accept()
    if stream not in active_connections:
        active_connections[stream] = []
    active_connections[stream].append(websocket)
    try:
        while True:
            # Keep the connection open and wait for messages (though we mostly push)
            await websocket.receive_text()
    except WebSocketDisconnect:
        active_connections[stream].remove(websocket)
        if not active_connections[stream]:
            del active_connections[stream]


@app.post("/publish/{stream:path}")
async def publish(stream: str, message: Message):
    disconnected = []

    # Log the data if it exists
    logs.log(stream, message.data)

    # Broadcast the message
    for connection in active_connections.get(stream, []):
        try:
            await connection.send_json(message.model_dump())
        except Exception:
            disconnected.append(connection)

    # Clean up stale connections
    for conn in disconnected:
        active_connections[stream].remove(conn)

    return {"status": "published", "listener_count": len(active_connections.get(stream, []))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=utils.FABRIC_PORT)