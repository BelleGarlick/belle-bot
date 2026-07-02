# Fabric

Fabric is the communications protocol that everything communicates over. This is akin to ROS. We can publish data to the fabric and subscribe to streams via WebSockets to get notified when messages are published.

## Usage

### Server
The fabric service runs on port 59990 by default. It provides a WebSocket endpoint for listening and a REST endpoint for publishing.

- `WS /listen/{stream}`: Connect to this WebSocket to receive messages published to the specified stream.
- `POST /publish/{stream}`: Send a JSON body with `{"data": {...}}` to publish to the stream.

### Client
Use the `FabricClient` in `belle_bot.fabric.client` to interact with the fabric.

## TODO
- Add testing
- Incorporate logging
- Replay events somehow - may have to just be done as and when needed