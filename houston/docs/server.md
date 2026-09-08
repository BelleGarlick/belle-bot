# Houston Server

The Houston server is a FastAPI application that serves as the central hub for data and management.

## Directory Structure

- `houston_server_api/`: Contains the FastAPI application and route definitions.
- `houston_server_core/`: Implements the core business logic, including replayers and models.
- `houston_server_gateways/`: Handles external integrations like file systems and SQLite.
- `houston_server_persistence/`: Manages data persistence and database models.

## How to Run

### 1. Prerequisites
- Python 3.x
- Dependencies installed from `requirements.txt` in the project root.

### 2. Running the API
You can use the provided script to start the server:

```bash
./houston/scripts/api.sh
```

Alternatively, manually:

```bash
export PYTHONPATH=houston/server
export HOUSTON_PATH=houston_data
python3 houston/server/houston_server_api/api.py
```

## Configuration

- `PYTHONPATH`: Must include `houston/server` for imports to work correctly.
- `HOUSTON_PATH`: Path to the directory where Houston stores its data (e.g., replays).

## API Documentation
Once the server is running, you can typically access the interactive Swagger UI at `http://localhost:8000/docs`.
