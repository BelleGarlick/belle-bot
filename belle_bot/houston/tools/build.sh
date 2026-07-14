#!/bin/bash
set -e

# Ensure the output directory exists
mkdir -p frontend/src/api

# Export the OpenAPI schema from the FastAPI app
# Using the requested houston/server/houston_server_api/api.py:api
PYTHONPATH=server python3 -c '
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

# Generate the TypeScript API using orval and fetch
# orval must be installed (e.g., via npm install -g orval)

cd frontend
npx orval --input ../openapi.json --output src/api/api.ts --client fetch

# Clean up
rm ../openapi.json

echo "API generated successfully at houston/frontend/api/api.ts"
