import os
import signal
import uuid
import json
import random
import socket
import subprocess
import time
from pathlib import Path

PID_FILE = "pid.txt"


"""
The houston server should be able to handle spawning subprocesses that allow for running replays 
to allow you to tap into or just to watch back. This module spawn the child processes needed to
make that work.

For this to work, the server spins up a custom fabric runtime which all the messages are 
communicated over, like normal. This allows components to interact with the system like 
they would during runtime. We also spin up a replayer component which downloads the replays from the 
houston server, and relays them through fabric as to re-simulate the events 
"""


# todo
#  send messages to the fabric server to pause and whatnot
#  have the replay server send messages to the fabric about it's replay data


def list_replayers():
    if not os.path.exists(PID_FILE):
        return []

    with open(PID_FILE, 'r') as pidfile:
        return json.load(pidfile)


def save_replayers(replayers):
    with open("pid.txt", 'w+') as pidfile:
        pidfile.write(json.dumps(replayers))


def is_port_in_use(port_number):
    """Returns True if the local port is currently occupied."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port_number))
            return False
        except OSError:
            return True


def start_process(name: str, replay_ids: list[str]):
    """
    Starts a process in the background and writes a PID file

    returns integer: pid
    """
    # first check which port to run it on
    while True:
        port = random.randint(10_000, 20_000)
        if not is_port_in_use(port):
            break

    path = Path(__file__).parent.parent.parent.parent.parent

    # then start the fabric server on that port
    common_env = {
        "FABRIC_PORT": str(port),
        "PYTHONPATH": str(path),
    }

    # Start a fabric process just for this replay
    fabric_process = subprocess.Popen(
        [
            f'{path}/.venv/bin/python',
            f'{path}/belle_bot/fabric/service.py',
        ],
        shell=False,
        env={
            **common_env,
            "REPLAYS": ','.join(replay_ids),
        }
    )

    # Start a replay process to run on the fabric server
    replay_process = subprocess.Popen(
        [
            f'{path}/.venv/bin/python',
            f'{path}/belle_bot/infra/replays/replayer.py',
        ],
        shell=False,
        env={
            **common_env,
            "REPLAYS": ','.join(replay_ids),
        }
    )

    data = {
        "replayer_id": str(uuid.uuid4()),
        "name": name,
        "port": port,
        "replay_ids": replay_ids,
        "start_time": time.time(),
        "fabric_pid": fabric_process.pid,
        "replay_pid": replay_process.pid
    }

    # Save to a pid file
    save_replayers(list_replayers() + [
        data
    ])

    return data


def stop_replayer(replayer_id: str):
    replayers = list_replayers()
    for replayer in replayers:
        if not replayer['replayer_id'] == replayer_id:
            continue

        os.kill(replayer['fabric_pid'], signal.SIGTERM)
        os.kill(replayer['replay_pid'], signal.SIGTERM)

    save_replayers([x for x in replayers if x['replayer_id'] != replayer_id])
