import json
import asyncio
import re
import sqlite3

from belle_bot.infra.fabric import FabricClient


"""
This module allows for replaying the given event
"""

# LOGS_DIR = "/Users/belle/Developer/belle-bot/replays/"
PATH = "/Users/belle/Developer/belle-bot/replays/c6d2c516-252c-4844-9743-40a293813ebe.txt"


PAGE_LENGTH = 100
CLIENT = FabricClient(host="0.0.0.0")
LOOP = True


def get_logs(limit: int = None, offset: int = None):
    with open(PATH, "r") as f:
        lines = f.readlines()

    split_lines = []
    for line in lines:
        split = line.split(",")

        key = split[0]
        time = split[1]
        remaining = ",".join(split[2:])

        split_lines.append({
            "stream": key,
            "timestamp": float(time),
            "value": json.loads(remaining)
        })

    return split_lines


async def main():
    page = 0
    last_item_time = None

    items = get_logs(limit=PAGE_LENGTH, offset=page * PAGE_LENGTH)
    while items:
        for item in items:
            current_item_time = float(item["timestamp"])

            if last_item_time is not None:
                wait_time = (current_item_time - last_item_time)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            last_item_time = current_item_time

            # Run this async
            asyncio.create_task(CLIENT.publish_async(item["stream"], item['value']))

        page += 1
        items = get_logs(limit=PAGE_LENGTH, offset=page * PAGE_LENGTH)

        if LOOP and page != 0 and len(items) == 0:
            page = 0
            items = get_logs(limit=PAGE_LENGTH, offset=page * PAGE_LENGTH)

    # Wait for all background tasks to finish
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())