import json
import asyncio
import os
import threading

from belle_bot.fabric import FabricClient
from belle_bot.houston.client.python.houston_api_client.api.replays import get_replay_file
from belle_bot.houston.client.python import houston_api_client

# todo create a common python houston client

houston_client = houston_api_client.Client(base_url="http://localhost:8081/")


"""
This module allows for replaying the given event
"""

REPLAY_IDS = os.environ["REPLAYS"].split(",")


PAGE_LENGTH = 100
CLIENT = FabricClient(host="0.0.0.0")
LOOP = True


class ReplayLoader(threading.Thread):

    def __init__(self, replay_id: str):
        threading.Thread.__init__(self)
        self.replay_id = replay_id
        self.events = []

    def run(self):
        response = get_replay_file.sync_detailed(replay_id=self.replay_id, client=houston_client)
        if response.status_code == 200:
            content_str = response.content.decode("utf-8")

            replay_data = []
            for line in content_str.split("\n"):
                split = line.split(",")
                if len(split) < 2:
                    continue

                key = split[0]
                time = split[1]
                remaining = ",".join(split[2:])

                replay_data.append({
                    "stream": key,
                    "timestamp": float(time),
                    "value": json.loads(remaining)
                })

            self.events = replay_data


class LogIterator:

    def __init__(self):
        self.idx = 0
        self.replay_idx = 0
        self.replay: ReplayLoader = ReplayLoader(REPLAY_IDS[0])
        self.next_replay: ReplayLoader | None = None

    def __iter__(self):
        self.idx = 0

        self.replay.run()

        return self

    def __next__(self):
        # If we're at the end of the replay, do teh ol' switcheroo
        if self.idx >= len(self.replay.events):
            self.idx = 0
            if self.next_replay is not None:
                self.replay, self.next_replay = self.next_replay, None
                self.replay.join()

        # Load and then increment the counter
        item = self.replay.events[self.idx]
        self.idx += 1

        # Start loading the next replay if needed
        if self.next_replay is None and len(REPLAY_IDS) > 1:
            self.replay_idx += 1
            next_replay = ReplayLoader(REPLAY_IDS[self.replay_idx % len(REPLAY_IDS)])
            next_replay.start()
            self.next_replay = next_replay

        return self.replay.replay_id, self.idx / len(self.replay.events), item


async def main():
    last_item_time = None
    last_update_time: float | None = None

    iterator = LogIterator()

    for replay_id, progress, item in iterator:
        current_item_time = float(item["timestamp"])

        if last_item_time is not None:
            wait_time = (current_item_time - last_item_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        last_item_time = current_item_time

        # Run this async
        asyncio.create_task(CLIENT.publish_async(item["stream"], item['value']))

        if last_update_time is None or current_item_time - last_update_time > 1:
            asyncio.create_task(CLIENT.publish_async("replayer", {
                "replay_id": item["replay_id"],
                "progress": progress
            }))


    # Wait for all background tasks to finish
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
