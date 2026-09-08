# Houston Python Client

The Houston Python client provides a simple interface for interacting with the Houston API from other Python-based components of the `belle-bot` project.

## Usage

The client is located in `houston/client/py`.

### Configuration
Most client functions require a `HoustonConfig` object which specifies the host and port of the Houston server.

```python
from houston.client.py.config import HoustonConfig

config = HoustonConfig(host="localhost", port=8000)
```

### Working with Replays
You can use the `replays` module to query and retrieve replay data.

```python
from houston.client.py import replays

# Query replays with pagination and optional tags
all_replays = replays.query_replays(config, page=0, tags=["training"])

# Get the contents of a specific replay file
replay_content = replays.get_replay_file(config, "replay_id_123")
```

## Internal Structure
- `config.py`: Defines the `HoustonConfig` class.
- `replays.py`: Contains functions for interacting with the replay-related API endpoints.
- `utils.py`: Contains helper functions for making HTTP requests (GET/POST).
