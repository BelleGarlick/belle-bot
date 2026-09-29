#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")"

echo "Running Ruff formatter..."
python3 -m ruff format houston

echo "Running Ruff linter (with auto-fix)..."
python3 -m ruff check houston --fix

echo "Done!"
