#!/bin/bash
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
HOUSTON_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
PROJECT_ROOT="$( cd "$HOUSTON_DIR/../.." && pwd )"

# Load NVM if it exists
export NVM_DIR="$HOME/.nvm"
if [ -s "$NVM_DIR/nvm.sh" ]; then
    . "$NVM_DIR/nvm.sh"
fi

# Set HOUSTON_PATH if it's not set
if [ -z "$HOUSTON_PATH" ]; then
    export HOUSTON_PATH="$PROJECT_ROOT/houston_data"
fi

# Ensure the output directory exists
mkdir -p "$HOUSTON_DIR/frontend/src/api"

# Export the OpenAPI schema from the FastAPI app
echo "Generating openapi.json..."
cd "$PROJECT_ROOT"
PYTHONPATH="$HOUSTON_DIR/server" .venv/bin/python -c '
import json
import sys
from houston_server_api.api import app

# Use "api" if available as requested, otherwise fallback to "app"
try:
    from houston_server_api.api import api
except ImportError:
    api = app

print(json.dumps(api.openapi()))
' > "$HOUSTON_DIR/openapi.json"

# Generate the TypeScript API using orval and fetch
echo "Generating TypeScript API..."
cd "$HOUSTON_DIR/frontend"
npx orval --config orval.config.ts

# Clean up
rm "$HOUSTON_DIR/openapi.json"

echo "API generated successfully"
