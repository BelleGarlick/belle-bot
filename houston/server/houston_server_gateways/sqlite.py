import json
import sqlite3
import os
from typing import Callable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
TReturn = TypeVar("TReturn")

REPLAYS_DB_PATH = os.environ.get("REPLAYS_DB_PATH", "houston.db")

def get_connection():
    conn = sqlite3.connect(REPLAYS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_table(table_name: str):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            pk TEXT PRIMARY KEY,
            data TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def put(table: str, pk: str, model: T) -> T:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"""
            INSERT INTO {table} (pk, data)
            VALUES (?, ?)
            ON CONFLICT(pk) DO UPDATE SET data=excluded.data
            """,
            (
                pk,
                model.model_dump_json()
            ),
        )
        conn.commit()
    finally:
        conn.close()

    return model


def get(table: str, pk, callback: Callable[[dict], TReturn]) -> TReturn:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT data FROM {table} WHERE pk = ?", (pk,))
        row = cursor.fetchone()
        if not row:
            return None

        return callback(json.loads(row[0]))
    finally:
        conn.close()


def query(
    table: str,
    page: int,
    callback: Callable[[dict], TReturn],
    page_size: int = 50,
) -> tuple[list[TReturn], int]:
    conn = get_connection()
    try:
        cursor = conn.cursor()

        sql_query = f"SELECT data FROM {table} LIMIT {page_size} OFFSET {page * page_size}"
        cursor.execute(sql_query)
        rows = cursor.fetchall()

        sql_query = f"SELECT count(*) FROM {table}"
        cursor.execute(sql_query)
        count, = cursor.fetchone()

        return [callback(row) for row in rows], count
    finally:
        conn.close()


def delete(table: str, pk: str):
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table} WHERE pk = ?", (pk,))
        conn.commit()
    finally:
        conn.close()
