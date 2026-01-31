from __future__ import annotations

from typing import Optional, Dict, List, Tuple, Literal

import polars as pl

from .constants import INTERVAL_MS
from .state import WatermarkState
from .binance_api import BinanceKlinesClient
from .storage import ParquetLake, _normalize_import_df
from .resample import resample_1m_to_target


class BinanceDatabase:
    def __init__(
        self,
        data_root: str = "data/binance",
        state_db: str = "data/binance_state.duckdb",
        timeout: int = 15,
        max_retries: int = 8,
    ):
        self.state = WatermarkState(state_db)
        self.client = BinanceKlinesClient(timeout=timeout, max_retries=max_retries)
        self.lake = ParquetLake(data_root)

    # ---- state ----
    def get_watermark(self, venue: str, symbol: str, interval: str) -> Optional[int]:
        return self.state.get(venue, symbol, interval)

    # ---- ingestion ----
    def sync(self, venue: str, symbol: str, interval: str, start_ms: int) -> Dict:
        last = self.state.get(venue, symbol, interval)
        cursor = int(start_ms) if last is None else int(last + INTERVAL_MS[interval])

        total = 0
        newest = last

        for batch in self.client.page_klines(venue, symbol, interval, cursor, None):
            df = self.lake.to_polars(batch, venue, symbol, interval)
            self.lake.write_partitioned(df)

            newest = int(df.select(pl.col("open_time").max()).item())
            self.state.set(venue, symbol, interval, newest)
            total += df.height

        return {"written": int(total), "last_open_time": newest}

    def backfill(self, venue: str, symbol: str, interval: str, start_ms: int, end_ms: int) -> Dict:
        total = 0
        newest = None

        for batch in self.client.page_klines(venue, symbol, interval, int(start_ms), int(end_ms)):
            df = self.lake.to_polars(batch, venue, symbol, interval)
            self.lake.write_partitioned(df)

            newest = int(df.select(pl.col("open_time").max()).item())
            total += df.height

        if newest is not None:
            last = self.state.get(venue, symbol, interval)
            if last is None or newest > last:
                self.state.set(venue, symbol, interval, newest)

        return {"written": int(total), "last_open_time": newest}

    # ---- query ----
    def query(
        self,
        venue: str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        *,
        as_: Literal["polars", "pandas"] = "polars",
        columns: Optional[List[str]] = None,
    ):
        return self.lake.query_scan(
            venue, symbol, interval, start_ms, end_ms,
            columns=columns,
            as_=as_,
        )

    # ---- gaps ----
    def ensure_gaps(self, venue: str, symbol: str, interval: str, start_ms: int, end_ms: int) -> Dict:
        df = self.query(
            venue, symbol, interval, start_ms, end_ms,
            as_="polars",
            columns=["open_time", "venue", "symbol", "interval"],
        )
        step = INTERVAL_MS[interval]

        if df.is_empty():
            r = self.backfill(venue, symbol, interval, start_ms, end_ms)
            return {"gaps": [(int(start_ms), int(end_ms))], "filled": [r]}

        ts = df.select(pl.col("open_time")).to_series().sort().unique().to_list()

        gaps: List[Tuple[int, int]] = []
        for a, b in zip(ts[:-1], ts[1:]):
            a = int(a); b = int(b)
            if b - a > step:
                gaps.append((a + step, b - step))

        filled = [self.backfill(venue, symbol, interval, gs, ge) for gs, ge in gaps]
        return {"gaps": gaps, "filled": filled}

    # ---- resample ----
    def resample_from_1m(self, venue: str, symbol: str, start_ms: int, end_ms: int, target_interval: str) -> Dict:
        base = self.query(
            venue, symbol, "1m", start_ms, end_ms,
            as_="polars",
            columns=[
                "open_time","open","high","low","close","volume",
                "quote_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol",
                "venue","symbol"
            ],
        )
        if base.is_empty():
            return {"written": 0}

        out = resample_1m_to_target(base, venue, symbol, target_interval)
        self.lake.write_partitioned(out)
        return {"written": int(out.height)}
    
    def import_klines(
        self,
        df,
        *,
        venue: str,
        symbol: str,
        interval: str,
        update_watermark: bool = True,
    ) -> dict:
        """
        Import external kline data into parquet lake.

        df: pandas or polars DataFrame
        """
        pl_df = _normalize_import_df(df, venue, symbol, interval)

        # 写入 parquet（append-only）
        self.lake.write_partitioned(pl_df)

        rows = pl_df.height
        max_open_time = int(pl_df.select(pl.col("open_time").max()).item())

        # 可选：推进 watermark
        if update_watermark:
            last = self.state.get(venue, symbol, interval)
            if last is None or max_open_time > last:
                self.state.set(venue, symbol, interval, max_open_time)

        return {
            "written": rows,
            "last_open_time": max_open_time,
            "venue": venue,
            "symbol": symbol,
            "interval": interval,
        }
    def compact_date_partition(
        self,
        venue: str,
        symbol: str,
        interval: str,
        date: str,
        *,
        target_rows_per_file: int = 2_000_000,
        delete_inputs: bool = True,
        force: bool = False,
    ) -> Dict:
        return self.lake.compact_date_partition(
            venue=venue,
            symbol=symbol,
            interval=interval,
            date=date,
            target_rows_per_file=target_rows_per_file,
            delete_inputs=delete_inputs,
            force=force,
        )

    def compact_range(
        self,
        venue: str,
        symbol: str,
        interval: str,
        dates: List[str],
        *,
        target_rows_per_file: int = 2_000_000,
        delete_inputs: bool = True,
        force: bool = False,
    ) -> Dict:
        results = []
        for d in dates:
            results.append(self.compact_date_partition(
                venue, symbol, interval, d,
                target_rows_per_file=target_rows_per_file,
                delete_inputs=delete_inputs,
                force=force,
            ))
        return {"results": results}
    
    def rebuild_watermark_from_parquet(
        self,
        venue: str,
        symbol: str,
        interval: str,
    ) -> Dict:
        m = self.lake.max_open_time(venue, symbol, interval)
        if m is None:
            return {"status": "no_local_data", "last_open_time": None}
        self.state.set(venue, symbol, interval, m)
        return {"status": "rebuilt", "last_open_time": m}

    def rebuild_state_for_many(
        self,
        items: List[tuple[str, str, str]],  # [(venue,symbol,interval), ...]
    ) -> Dict:
        out = []
        for v, s, i in items:
            out.append({"venue": v, "symbol": s, "interval": i, **self.rebuild_watermark_from_parquet(v, s, i)})
        return {"results": out}