#!/bin/bash
set -e

export PYTHONPATH=houston/server
export HOUSTON_PATH=/run/media/belle/Houston

python3 houston/server/houston_server_api/api.py
