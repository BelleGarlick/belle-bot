#!/bin/bash
set -e

export PYTHONPATH=server
export HOUSTON_PATH=houston_data

python3 server/houston_server_api/api.py
