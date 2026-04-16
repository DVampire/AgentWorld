import time
from datetime import datetime, timezone, timedelta

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl

from BinanceDatabase.src.core import BacktestBinanceDatabase


def now_ms() -> int:
    return int(time.time() * 1000)


def floor_to_minute_ms(ts_ms: int) -> int:
    return (ts_ms // 60_000) * 60_000


def utc_ms(dt_utc: datetime) -> int:
    return int(dt_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)


def main():
    svc = BacktestBinanceDatabase(
        data_root="data/binance",
        state_db="data/binance_state.duckdb",
        timeout=15,
        max_retries=8,
    )

    venue = "binance_um"
    symbol = "BTCUSDT"
    interval = "1m"


    svc.rebuild_watermark_from_parquet("binance_um", "BTCUSDT", "1m")
    
    start_ms = utc_ms(datetime(2021, 4, 13))
    end_closed_open = floor_to_minute_ms(now_ms()) - 60_000

    end_ms = end_closed_open

    df_1m = svc.query(venue, symbol, interval, start_ms, end_ms, as_="polars")

    try:
        n_unique_before = df_1m.select(pl.col("open_time")).n_unique()
    except Exception:
        n_unique_before = 0

    r_sync = svc.sync(venue, symbol, interval, start_ms)
    print("sync:", r_sync)

    df_1m_after = svc.query(
        venue, symbol, interval, start_ms, end_ms,
        as_="polars", columns=["open_time", "venue", "symbol", "interval","open","high","low","close","volume"],
    )
    n_unique_after = df_1m_after.select(pl.col("open_time")).n_unique()
    print("unique open_time before:", n_unique_before, "after:", n_unique_after)
    print(df_1m_after.head(20))
    


if __name__ == "__main__":
    main()