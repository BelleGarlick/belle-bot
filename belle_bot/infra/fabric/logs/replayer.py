import json
import os
import asyncio

from belle_bot.infra.fabric import FabricClient

from belle_bot.infra.fabric.logs import get_logs


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

# def get_logs(limit: int = None, offset: int = None):
#     log_dir = Path(DEFAULT_LOG_DIR)
#
#     if not log_dir.exists():
#         return []
#
#     # Find all .db files in the log directory
#     chunk_files = list(log_dir.glob("*.db"))
#
#     all_rows = []
#     for path in chunk_files:
#         try:
#             conn = sqlite3.connect(path)
#             conn.row_factory = sqlite3.Row
#             cursor = conn.cursor()
#
#             # Check if table exists
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='service_logs'")
#             if not cursor.fetchone():
#                 conn.close()
#                 continue
#
#             query = "SELECT * FROM service_logs"
#             cursor.execute(query)
#             rows = cursor.fetchall()
#             all_rows.extend([dict(row) for row in rows])
#             conn.close()
#         except sqlite3.Error:
#             continue
#
#     # Sort all rows by timestamp since files are UUID-named and might not be in order
#     all_rows.sort(key=lambda x: float(x["timestamp"]))
#
#     # Apply limit and offset
#     if offset is not None:
#         all_rows = all_rows[offset:]
#     if limit is not None:
#         all_rows = all_rows[:limit]
#
#     return all_rows


if __name__ == "__main__":
    asyncio.run(main())