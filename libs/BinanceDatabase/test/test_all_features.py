import os
import shutil
import time
from pathlib import Path
from datetime import datetime, timezone

import polars as pl
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from BinanceDatabase.src.core import BacktestBinanceDatabase as BacktestBinanceData


def now_ms() -> int:
    return int(time.time() * 1000)


def floor_to_minute_ms(ts_ms: int) -> int:
    return (ts_ms // 60_000) * 60_000


def ms_to_utc_date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()


def tree(root: Path, max_files: int = 60) -> None:
    if not root.exists():
        print(f"[tree] {root} (missing)")
        return
    files = [p for p in root.rglob("*") if p.is_file()]
    print(f"[tree] {root} total_files={len(files)}")
    for p in files[:max_files]:
        print(" -", p.as_posix())
    if len(files) > max_files:
        print(f" ... ({len(files)-max_files} more)")


def assert_nonempty_df(df: pl.DataFrame, msg: str):
    assert isinstance(df, pl.DataFrame), "Expected polars.DataFrame"
    assert df.height > 0, msg


def assert_has_cols(df: pl.DataFrame, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"


def run():
    # -------------------------
    # 0) Sandbox paths
    # -------------------------
    base_dir = Path("tmp_test_backtest_binance_data_all")
    data_root = base_dir / "data" / "binance"
    state_db = base_dir / "data" / "binance_state.duckdb"

    if base_dir.exists():
        shutil.rmtree(base_dir)
    data_root.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1) Init service
    # -------------------------
    svc = BacktestBinanceData(
        data_root=str(data_root),
        state_db=str(state_db),
        timeout=15,
        max_retries=8,
    )

    # Choose venue/symbol/interval
    venue = "binance_um"  # change to binance_spot if you prefer
    symbol = "BTCUSDT"
    interval = "1m"

    # Use a "closed-bar-safe" end: last closed minute open_time
    end_closed_open = floor_to_minute_ms(now_ms()) - 60_000
    # Build a 3-hour window to make sure we have enough data
    start_ms = end_closed_open - 3 * 60 * 60 * 1000
    # For backfill end_ms we can use close_time-ish; but our query filters by open_time
    # So here set end_ms as end_closed_open (inclusive on open_time)
    end_ms = end_closed_open

    print("=== 0) Initial state ===")
    print("state_db exists:", state_db.exists())
    print("watermark:", svc.get_watermark(venue, symbol, interval))
    tree(base_dir)

    # -------------------------
    # 2) backfill
    # -------------------------
    print("\n=== 1) backfill (3 hours, closed bars) ===")
    r_backfill = svc.backfill(venue, symbol, interval, start_ms, end_ms)
    print("backfill:", r_backfill)

    print("\n=== 2) verify DuckDB created + watermark updated ===")
    print("state_db exists:", state_db.exists())
    wm = svc.get_watermark(venue, symbol, interval)
    print("watermark:", wm)
    assert state_db.exists(), "DuckDB file should exist after backfill/sync"
    assert wm is not None, "watermark should not be None after write"

    # -------------------------
    # 3) query (polars, full)
    # -------------------------
    print("\n=== 3) query 1m (polars) ===")
    df_1m = svc.query(venue, symbol, interval, start_ms, end_ms, as_="polars")
    print(df_1m.dtypes)
    print("rows:", df_1m.height, "cols:", len(df_1m.columns))
    print(df_1m.head(3))
    assert_nonempty_df(df_1m, "Expected non-empty 1m data after backfill")
    assert_has_cols(df_1m, ["open_time", "open", "high", "low", "close", "volume", "date"])

    # -------------------------
    # 4) sync (should be near-zero net-new for same time window)
    #    Note: sync writes up to "now", so written can be >0, but net unique in [start,end] should stay 0
    # -------------------------
    print("\n=== 4) sync (validate net-new within the SAME [start,end] window) ===")
    n_unique_before = df_1m.select(pl.col("open_time")).n_unique()

    r_sync = svc.sync(venue, symbol, interval, start_ms)
    print("sync:", r_sync)

    df_1m_after = svc.query(
        venue, symbol, interval, start_ms, end_ms,
        as_="polars", columns=["open_time", "venue", "symbol", "interval"]
    )
    n_unique_after = df_1m_after.select(pl.col("open_time")).n_unique()
    print("unique open_time before:", n_unique_before, "after:", n_unique_after)
    assert n_unique_after == n_unique_before, "Within the fixed window, unique bars should not increase"

    # -------------------------
    # 5) resample 1h + query
    # -------------------------
    print("\n=== 5) resample_from_1m -> 1h, then query 1h ===")
    r_rs = svc.resample_from_1m(venue, symbol, start_ms, end_ms, target_interval="1h")
    print("resample:", r_rs)

    df_1h = svc.query(venue, symbol, "1h", start_ms, end_ms, as_="polars")
    print("1h rows:", df_1h.height)
    print(df_1h.head(5))
    assert_nonempty_df(df_1h, "Expected non-empty 1h data after resample")
    assert_has_cols(df_1h, ["open_time", "open", "high", "low", "close", "volume"])

    # -------------------------
    # 6) ensure_gaps (simulate by deleting one date partition under 1m)
    # -------------------------
    print("\n=== 6) ensure_gaps (simulate missing date partition) ===")
    # identify a date partition to delete that overlaps our window
    date0 = ms_to_utc_date(start_ms)
    date1 = ms_to_utc_date(end_ms)

    base_path = data_root / f"venue={venue}" / f"symbol={symbol}" / f"interval={interval}"
    victim = None
    for d in [date0, date1]:
        cand = base_path / f"date={d}"
        if cand.exists():
            victim = cand
            break

    if victim is None:
        print("No date partition found to delete; skipping gap simulation.")
    else:
        # record count before deletion
        pre = svc.query(venue, symbol, interval, start_ms, end_ms, as_="polars", columns=["open_time"])
        pre_n = pre.height
        print("rows before delete:", pre_n)

        print("deleting:", victim.as_posix())
        shutil.rmtree(victim)

        mid = svc.query(venue, symbol, interval, start_ms, end_ms, as_="polars", columns=["open_time"])
        mid_n = mid.height
        print("rows after delete:", mid_n)
        assert mid_n <= pre_n, "After deleting a partition, rows should not increase"

        r_gap = svc.ensure_gaps(venue, symbol, interval, start_ms, end_ms)
        print("ensure_gaps:", r_gap)

        post = svc.query(venue, symbol, interval, start_ms, end_ms, as_="polars", columns=["open_time"])
        post_n = post.height
        print("rows after ensure_gaps:", post_n)
        assert post_n >= mid_n, "Gap repair should not reduce rows further"

    # -------------------------
    # 7) import_klines (import a small synthetic dataset into a new venue)
    #    This tests "allow import data" without interfering with Binance data.
    # -------------------------
    print("\n=== 7) import_klines (synthetic) into venue=import_test ===")
    # Create 10 minutes synthetic bars
    import_venue = "import_test"
    import_symbol = "BTCUSDT"
    import_interval = "1m"

    t0 = floor_to_minute_ms(now_ms()) - 60_000 * 30
    times = [t0 + i * 60_000 for i in range(10)]

    imp = pl.DataFrame({
        "open_time": times,
        "open": [100.0 + i for i in range(10)],
        "high": [100.5 + i for i in range(10)],
        "low": [99.5 + i for i in range(10)],
        "close": [100.2 + i for i in range(10)],
        "volume": [1.0] * 10,
        "close_time": [t + 60_000 - 1 for t in times],
        "quote_volume": [10.0] * 10,
        "num_trades": [1] * 10,
        "taker_buy_base_vol": [0.5] * 10,
        "taker_buy_quote_vol": [5.0] * 10,
    })

    # If your BacktestBinanceData already has import_klines implemented:
    r_imp = svc.import_klines(
        imp,
        venue=import_venue,
        symbol=import_symbol,
        interval=import_interval,
        update_watermark=True,
    )
    print("import:", r_imp)

    df_imp = svc.query(import_venue, import_symbol, import_interval, t0, t0 + 9 * 60_000, as_="polars")
    print("imported rows:", df_imp.height)
    print(df_imp.head(3))
    assert df_imp.height == 10, "imported dataset should have 10 rows"

    # -------------------------
    # 8) compact (compact a date partition for imported data)
    # -------------------------
    print("\n=== 8) compact_date_partition for import_test ===")
    d_imp = ms_to_utc_date(t0)
    r_comp = svc.compact_date_partition(
        venue=import_venue,
        symbol=import_symbol,
        interval=import_interval,
        date=d_imp,
        target_rows_per_file=2_000_000,
        delete_inputs=True,
        force=True,
    )
    print("compact:", r_comp)

    # Ensure still readable after compact
    df_imp2 = svc.query(import_venue, import_symbol, import_interval, t0, t0 + 9 * 60_000, as_="polars")
    assert df_imp2.height == 10, "data should remain after compact"
    assert df_imp2.select("open_time").to_series().is_sorted(), "open_time should be sorted after compact"

    # -------------------------
    # 9) final tree
    # -------------------------
    print("\n=== 9) Final tree ===")
    tree(base_dir)

    print("\nALL FEATURE TESTS PASSED")


if __name__ == "__main__":
    run()
