#!/bin/bash
set -e

# Load NVM if it exists
export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    . "$NVM_DIR/nvm.sh"
fi

export HOUSTON_PATH=

# Ensure the output directory exists
mkdir -p frontend/src/api

# Export the OpenAPI schema from the FastAPI app
# Using the requested houston/server/houston_server_api/api.py:api
PYTHONPATH=server /Users/belle/Developer/belle-bot/.venv/bin/python -c '
import json
import sys
from houston_server_api.api import app

# Use "api" if available as requested, otherwise fallback to "app"
try:
    from houston_server_api.api import api
except ImportError:
    api = app

print(json.dumps(api.openapi()))
' > openapi.json

# Generate the Python client using openapi-python-client
PATH="/Users/belle/Developer/belle-bot/.venv/bin:$PATH" openapi-python-client generate --path openapi.json --output-path client/python --overwrite

# Generate the TypeScript API using orval and fetch
# orval must be installed (e.g., via npm install -g orval)

cd frontend
npx orval --config orval.config.ts

# Clean up
#rm ../openapi.json

echo "API generated successfully at houston/frontend/api/api.ts"
