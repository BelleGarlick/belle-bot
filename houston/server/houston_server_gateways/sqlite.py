import json
import os
import sqlite3
import typing
from typing import Callable, TypeVar

from pydantic import BaseModel

from houston_server_gateways.utils import get_houston_data_root

T = TypeVar("T", bound=BaseModel)
TReturn = TypeVar("TReturn")

REPLAYS_DB_PATH = get_houston_data_root() / "houston.db"


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
            (pk, model.model_dump_json()),
        )
        conn.commit()
    finally:
        conn.close()

    return model


def get(table: str, pk: str, callback: Callable[[dict], TReturn]) -> TReturn | None:
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
    tags: list[str] | None = None,
    match_all_tags: bool = True,
) -> tuple[list[TReturn], int]:
    conn = get_connection()
    try:
        cursor = conn.cursor()
        offset = max(0, page) * page_size

        clean_tags = [str(t).strip() for t in (tags or []) if str(t).strip()]

        if clean_tags:
            placeholders = ",".join(["?"] * len(clean_tags))

            if match_all_tags:
                # Requires record to match ALL tags in clean_tags
                where_clause = f"""
                    json_type({table}.data, '$.tags') = 'array'
                    AND (
                        SELECT COUNT(DISTINCT json_each.value)
                        FROM json_each({table}.data, '$.tags')
                        WHERE json_each.type = 'text' 
                          AND json_each.value IN ({placeholders})
                    ) = {len(clean_tags)}
                """
            else:
                # Requires record to match AT LEAST ONE tag in clean_tags
                where_clause = f"""
                    json_type({table}.data, '$.tags') = 'array'
                    AND EXISTS (
                        SELECT 1 
                        FROM json_each({table}.data, '$.tags') 
                        WHERE json_each.type = 'text' 
                          AND json_each.value IN ({placeholders})
                    )
                """

            sql_query = f"""
                SELECT data 
                FROM {table}
                WHERE {where_clause}
                LIMIT ? OFFSET ?
            """
            cursor.execute(sql_query, (*clean_tags, page_size, offset))
            rows = cursor.fetchall()

            sql_count_query = f"""
                SELECT COUNT(*) 
                FROM {table}
                WHERE {where_clause}
            """
            cursor.execute(sql_count_query, clean_tags)
            count = cursor.fetchone()[0]
        else:
            sql_query = f"SELECT data FROM {table} LIMIT ? OFFSET ?"
            cursor.execute(sql_query, (page_size, offset))
            rows = cursor.fetchall()

            sql_count_query = f"SELECT count(*) FROM {table}"
            cursor.execute(sql_count_query)
            count = cursor.fetchone()[0]

        return [callback(json.loads(row["data"])) for row in rows], count
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