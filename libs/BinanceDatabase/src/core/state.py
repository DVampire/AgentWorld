from __future__ import annotations

import os
from typing import Optional
import duckdb


class WatermarkState:
    def __init__(self, state_db: str):
        self.state_db = state_db
        os.makedirs(os.path.dirname(self.state_db) or ".", exist_ok=True)
        self._init_db()

    def _conn(self):
        return duckdb.connect(self.state_db)

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS ingest_state (
              venue VARCHAR,
              symbol VARCHAR,
              interval VARCHAR,
              last_open_time BIGINT,
              updated_at TIMESTAMP,
              PRIMARY KEY (venue, symbol, interval)
            );
            """)

    def get(self, venue: str, symbol: str, interval: str) -> Optional[int]:
        with self._conn() as c:
            r = c.execute(
                "SELECT last_open_time FROM ingest_state WHERE venue=? AND symbol=? AND interval=?",
                [venue, symbol, interval],
            ).fetchone()
            return int(r[0]) if r else None

    def set(self, venue: str, symbol: str, interval: str, ts: int) -> None:
        with self._conn() as c:
            c.execute("""
            INSERT INTO ingest_state (venue, symbol, interval, last_open_time, updated_at)
            VALUES (?, ?, ?, ?, now())
            ON CONFLICT (venue, symbol, interval)
            DO UPDATE SET last_open_time=excluded.last_open_time, updated_at=excluded.updated_at
            """, [venue, symbol, interval, int(ts)])
