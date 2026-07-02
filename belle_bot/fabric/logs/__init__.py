import json
import os
import sqlite3
import time

FABRIC_PATH = os.environ.get("FABRIC_PATH")


def initialise():
    if not FABRIC_PATH:
        return

    conn = sqlite3.connect(FABRIC_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS service_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            value BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log(service_name: str, data: dict[str, str]):
    if not FABRIC_PATH:
        return

    conn = sqlite3.connect(FABRIC_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO service_logs (service_name, timestamp, value) VALUES (?, ?, ?)",
        (service_name, str(time.time()), json.dumps(data))
    )

    conn.commit()
    conn.close()


def get_logs(limit: int = None, offset: int = None):
    if not FABRIC_PATH:
        return []

    conn = sqlite3.connect(FABRIC_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM service_logs"
    params = []

    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
        if offset is not None:
            query += " OFFSET ?"
            params.append(offset)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]
