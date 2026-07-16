import os
import signal
import sys
import json
import subprocess
from pathlib import Path

import psutil

path = Path(__file__)
PROCESSES_FILE = path.parent / "processes.json"
belle_bot_dir = path.parent.parent

if belle_bot_dir.name != "belle_bot":
    raise Exception("This script must be run from belle_bot")

# Path to the central log file in the project root
LOG_FILE = path.parent / "belle_bot.log"



def get_valid_components(filter_key: str) -> list[dict]:
    """
    Finds and returns a list of component definitions that match the filter key.

    Searches for 'component.json' files within the belle_bot directory,
    excluding 'node_modules'. Components are included if their name
    starts with the prefix derived from the filter_key.

    Args:
        filter_key: A string used to filter components by name. Supports '*' as a wildcard.

    Returns:
        A list of dictionaries, each representing a component's definition.
    """
    valid_key_definitions = []

    # Calculate the prefix filter form the arg
    prefix = filter_key.split("*")[0]

    for path, _, files in os.walk(belle_bot_dir):
        if "node_modules" in path:
            continue

        if "component.json" in files:
            with open(os.path.join(path, "component.json"), "r") as f:
                component_definition = json.load(f)

            component_definition['location'] = path

            if component_definition['name'].startswith(prefix):
                valid_key_definitions.append((component_definition))

    return valid_key_definitions


def start_processes(existing_processes: dict[str, int], filter_key: str) -> None:
    """
    Starts components that match the filter key and are not already running.

    Launched processes have their stdout and stderr redirected to LOG_FILE.
    The log file is wiped at the start of the process.
    The process IDs are saved to PROCESSES_FILE.

    Args:
        existing_processes: A dictionary mapping component names to their PIDs.
        filter_key: A string used to filter which components to start.
    """
    # Open the central log file in write mode to wipe it
    with open(LOG_FILE, "w") as log_file:
        for definition in get_valid_components(filter_key):
            # If this is component is already running then don't run another
            if definition['name'] in existing_processes:
                print(definition['name'] + " is already running.")
                continue

            entry_point = os.path.join(definition['location'], definition['entry'])

            # todo verify the runtime type

            # Prepare environment variables
            env = os.environ.copy()
            if 'environ' in definition:
                # Ensure all environment variable values are strings
                stringified_environ = {k: str(v) for k, v in definition['environ'].items()}
                env.update(stringified_environ)

            # Launch the component and pipe outputs to the central log file
            process = subprocess.Popen(
                [sys.executable, entry_point],
                env=env,
                cwd=definition['location'],
                stdout=log_file,
                stderr=log_file,
                start_new_session=True
            )

            print("Launched " + definition['name'])

            # Save the created process to the list of items and save the file
            known_processes[definition['name']] = process.pid
            with open(PROCESSES_FILE, "w+") as f:
                json.dump(known_processes, f)


def existing_known_processes() -> dict[str, int]:
    """
    Loads and returns the dictionary of known processes from the PROCESSES_FILE.

    Returns:
        A dictionary mapping component names to their PIDs. Returns an empty dict
        if the file does not exist.
    """
    if os.path.exists(PROCESSES_FILE):
        with open(PROCESSES_FILE, "r") as f:
            return json.load(f)

    return {}


def filter_known_processes(known_processes: dict[str, int]) -> dict[str, int]:
    """
    Removes processes from the known_processes dictionary that are no longer running.

    Args:
        known_processes: A dictionary mapping component names to their PIDs.

    Returns:
        The updated dictionary containing only the processes that are still active.
    """
    for key, pid in list(known_processes.items()):
        if not psutil.pid_exists(pid):
            del known_processes[key]

    return known_processes


def stop_processes(known_processes: dict[str, int], filter_key: str) -> None:
    """
    Stops running components that match the filter key.

    Sends a SIGTERM signal to the process and removes it from the known_processes
    dictionary and the PROCESSES_FILE.

    Args:
        known_processes: A dictionary mapping component names to their PIDs.
        filter_key: A string used to filter which components to stop.
    """
    for definition in get_valid_components(filter_key):
        if definition['name'] in known_processes:
            pid = known_processes[definition['name']]

            print("Killing " + definition['name'])
            os.kill(pid, signal.SIGTERM)

            del known_processes[definition['name']]

            with open(PROCESSES_FILE, "w+") as f:
                json.dump(known_processes, f)

    print("Done.")


def list_processes(known_processes: dict[str, int], filter_key: str) -> None:
    """
    Prints a table of currently running processes that match the filter key.

    Args:
        known_processes: A dictionary mapping component names to their PIDs.
        filter_key: A string used to filter which processes to list.
    """
    prefix = filter_key.split("*")[0]
    filtered_processes = {k: v for k, v in known_processes.items() if k.startswith(prefix)}

    if not filtered_processes:
        print("No processes found matching the key.")
        return

    padd_length = max([len(x) for x in list(filtered_processes)]) + 2

    print(f"{'Name'.ljust(padd_length)}| PID")
    for key, pid in list(filtered_processes.items()):
        print(f"{key.ljust(padd_length)}| {pid}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m belle_bot.launchpad <start|stop|list> <key>")
        sys.exit(1)

    action = sys.argv[1].lower()
    command_key = sys.argv[2]

    known_processes = existing_known_processes()
    known_processes = filter_known_processes(known_processes)

    if action == "start":
        start_processes(known_processes, command_key)
    elif action == "stop":
        stop_processes(known_processes, command_key)
    elif action == "list":
        list_processes(known_processes, command_key)
    else:
        print(f"Unknown action: {action}")
        print("Usage: python -m belle_bot.launchpad <start|stop|list> <key>")
        sys.exit(1)
