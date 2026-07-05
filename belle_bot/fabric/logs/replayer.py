import json
import os
import asyncio

from belle_bot.fabric import FabricClient

os.environ["FABRIC_PATH"] = "/Users/belle/Developer/belle-bot/event3.db"

from belle_bot.fabric.logs import get_logs


"""
This module allows for replaying the given event
"""


PAGE_LENGTH = 100
CLIENT = FabricClient()
LOOP = True

async def main():
    page = 0
    last_item_time = None

    items = get_logs(limit=PAGE_LENGTH, offset=page * PAGE_LENGTH)
    while items:
        for item in items:
            current_item_time = float(item["timestamp"])
            
            if last_item_time is not None:
                wait_time = current_item_time - last_item_time
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            
            last_item_time = current_item_time

            # Run this async
            asyncio.create_task(CLIENT.publish_async(item["service_name"], json.loads(item['value'])))

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