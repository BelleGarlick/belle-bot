from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request

from belle_bot.infra.fabric import utils

app = FastAPI()

# In-memory storage for active websocket connections: stream_name -> list of WebSocket
active_connections: dict[str, list[WebSocket]] = {}


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
async def publish(stream: str, message: Request):
    targets = []
    for key in active_connections:
        key_prefix = key.split("*")[0]
        if stream.startswith(key_prefix):
            targets += active_connections[key]

    body = await message.body()
    body = body.decode("utf-8")

    # Broadcast the message
    disconnected = []
    for connection in targets:
        try:
            await connection.send_text(body)
        except Exception as e:
            print(e)
            disconnected.append(connection)

    # Clean up stale connections
    for conn in disconnected:
        active_connections[stream].remove(conn)

    return {"status": "published", "listener_count": len(active_connections.get(stream, []))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=utils.FABRIC_PORT)