import traceback

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
            await websocket.receive()

    except WebSocketDisconnect:
        print(f"Client disconnected from stream: {stream}")
    except Exception:
        traceback.print_exc()
    finally:
        # Cleanup guarantees this runs even if an unexpected error occurs
        if websocket in active_connections.get(stream, []):
            active_connections[stream].remove(websocket)
            if not active_connections[stream]:
                del active_connections[stream]


@app.post("/publish/{stream:path}")
async def publish(stream: str, message: Request):
    targets = []
    for key, conns in list(active_connections.items()):
        key_prefix = key.split("*")[0]
        if stream.startswith(key_prefix):
            targets += conns

    body = await message.body()
    body = body.decode("utf-8")

    dead = []
    for connection in targets:
        try:
            await connection.send_text(body)
        except Exception:
            dead.append(connection)

    for connection in dead:
        for stream_name, conns in list(active_connections.items()):
            if connection in conns:
                conns.remove(connection)
                if not conns:
                    del active_connections[stream_name]

    return {"status": "published", "listener_count": len(active_connections.get(stream, []))}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=utils.FABRIC_PORT, log_level="info")