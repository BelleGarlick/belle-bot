import json
import asyncio
import sqlite3

from belle_bot.infra.fabric import FabricClient


"""
This module allows for replaying the given event
"""

# LOGS_DIR = "/Users/belle/Developer/belle-bot/replays/"
PATH = "/Users/belle/Developer/belle-bot/replays/f522277d-ef84-4327-9ed1-26a4017ff32e.db"


PAGE_LENGTH = 100
CLIENT = FabricClient()
LOOP = True


def get_logs(limit: int = None, offset: int = None):
    all_rows = []

    try:
        conn = sqlite3.connect(PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='service_logs'")
        if not cursor.fetchone():
            conn.close()

        query = "SELECT * FROM service_logs"
        cursor.execute(query)
        rows = cursor.fetchall()
        all_rows.extend([dict(row) for row in rows])
        conn.close()
    except sqlite3.Error:
        pass

    # Sort all rows by timestamp since files are UUID-named and might not be in order
    all_rows.sort(key=lambda x: float(x["timestamp"]))

    # Apply limit and offset
    if offset is not None:
        all_rows = all_rows[offset:]
    if limit is not None:
        all_rows = all_rows[:limit]

    return all_rows


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
                    await asyncio.sleep(wait_time / 4)
            
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