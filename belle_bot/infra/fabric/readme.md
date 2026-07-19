# Fabric

Fabric is the communications protocol that everything communicates over. This is akin to ROS. We can publish data to the fabric and subscribe to streams via WebSockets to get notified when messages are published. 

As data is sent over the fabric, a logger may be configured which allows for all the events in the system to be stored to disk. Doing so will allow for replaying the whole system at a later date to understand what the system is doing / re-simulate events. This will allow us to at a later date train on and improve on the data caught in the fabric.

## Running the Fabric Service

The Fabric service is a FastAPI application that acts as the central hub for all communications.

### Basic Run (Without Logging)
To run the fabric without persistent logging:
```bash
python3 belle_bot/infra/fabric/service.py
```

### Running with Logging
To enable logging of all messages to a SQLite database, set the `SAVE_REPLAY_ENABLED` environment variable to `true`:
```bash
SAVE_REPLAY_ENABLED=true python3 belle_bot/infra/fabric/service.py
```
Logs will be stored in 10-minute chunks in the `replays/` directory. Each chunk file is named with a UUIDv4.

## Usage

### Server
The fabric service runs on port 59990 by default. It provides a WebSocket endpoint for listening and a REST endpoint for publishing.

- `WS /listen/{stream}`: Connect to this WebSocket to receive messages published to the specified stream.
- `POST /publish/{stream}`: Send a JSON body with `{"data": {...}}` to publish to the stream.

### Client
Use the `FabricClient` in `belle_bot.fabric.client` to interact with the fabric programmatically.

```python
from belle_bot.infra.fabric import FabricClient

client = FabricClient()

client.publish("my_stream", {"key": "value"})
client.listen("another_stream", print)
```

### Replaying Logs
You can use the replayer tool in `belle_bot/infra/fabric/logs/replayer.py` to play back events stored in the `replays/` directory.
