import json
import asyncio
import os
import threading

from belle_bot.fabric import FabricClient
from belle_bot.houston.client.py import replays

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
        replay_data = []

        lines = replays.get_replay_file(self.replay_id).split("\n")
        for line in lines:
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
        await CLIENT.publish_async(item["stream"], item['value'])

        if last_update_time is None or current_item_time - last_update_time > 1:
            last_update_time = current_item_time
            await CLIENT.publish_async("replayer", {
                "replay_id": replay_id,
                "progress": progress
            })

    # Wait for all background tasks to finish
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
