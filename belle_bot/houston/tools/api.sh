#!/bin/bash
set -e

export PYTHONPATH=belle_bot/houston/server
export HOUSTON_PATH=houston_data

python3 belle_bot/houston/server/houston_server_api/api.py
