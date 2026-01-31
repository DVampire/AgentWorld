import os
import glob
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import polars as pl
import time
from src.backtest_binance_data import BacktestBinanceData

# 你当前的 interval->ms 映射若在包内可导入就导入；否则在此维护一致版本
INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "6h": 21_600_000, "8h": 28_800_000,
    "12h": 43_200_000, "1d": 86_400_000,
}
def now_ms() -> int:
    return int(time.time() * 1000)

def _glob_parquets(data_root: str, venue: str, symbol: str, interval: str) -> List[str]:
    base = os.path.join(data_root, f"venue={venue}", f"symbol={symbol}", f"interval={interval}")
    if not os.path.exists(base):
        return []
    return glob.glob(os.path.join(base, "**", "*.parquet"), recursive=True)


def _scan_ot(files: List[str]) -> pl.LazyFrame:
    # 只扫描 open_time（最轻）
    return pl.scan_parquet(files).select(pl.col("open_time").cast(pl.Int64).alias("open_time"))


def _scan_daily_counts(files: List[str]) -> pl.LazyFrame:
    # 扫 open_time 并转日期，做每日条数统计
    return (
        pl.scan_parquet(files)
        .select(pl.col("open_time").cast(pl.Int64).alias("open_time"))
        .with_columns(pl.from_epoch("open_time", time_unit="ms").dt.date().alias("date"))
        .group_by("date")
        .agg(pl.len().alias("rows"))
        .sort("date")
    )


def _expected_rows_per_day(interval: str) -> Optional[int]:
    if interval not in INTERVAL_MS:
        return None
    day_ms = 86_400_000
    step = INTERVAL_MS[interval]
    if day_ms % step != 0:
        return None
    return day_ms // step


def _find_gaps_from_sorted_times(sorted_times: List[int], step: int, max_report: int = 20) -> List[Tuple[int, int]]:
    """
    Given strictly increasing unique open_time list, find missing segments.
    Returns list of (gap_start_open_time, gap_end_open_time) inclusive in bar units.
    """
    gaps = []
    for a, b in zip(sorted_times[:-1], sorted_times[1:]):
        if b - a > step:
            gaps.append((a + step, b - step))
            if len(gaps) >= max_report:
                break
    return gaps


def check_data_health(
    svc: BacktestBinanceData,
    *,
    venue: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    severity_row_threshold: float = 0.98,
    max_gap_report: int = 20,
    small_file_threshold_bytes: int = 1_000_000,
    warn_small_files_count: int = 50,
    warn_parts_per_day: int = 30,
) -> Dict[str, Any]:
    """
    Returns a structured report dict.
    - severity_row_threshold: daily rows < expected*threshold => warn
    """

    report: Dict[str, Any] = {
        "target": {"venue": venue, "symbol": symbol, "interval": interval, "start_ms": start_ms, "end_ms": end_ms},
        "storage": {},
        "watermark": {},
        "stats": {},
        "checks": {},
        "recommendations": [],
    }

    files = _glob_parquets(svc.lake.data_root if hasattr(svc, "lake") else svc.data_root, venue, symbol, interval)
    report["storage"]["parquet_files"] = len(files)

    if not files:
        report["checks"]["exists"] = False
        report["recommendations"].append("No parquet files found for this (venue,symbol,interval). Run backfill/sync first.")
        return report

    # File-level health (small files / partition spread)
    sizes = []
    parts_per_date: Dict[str, int] = {}
    for f in files:
        try:
            sz = os.path.getsize(f)
        except OSError:
            sz = 0
        sizes.append(sz)

        # derive date=YYYY-MM-DD from path
        # .../date=YYYY-MM-DD/xxx.parquet
        segs = f.replace("\\", "/").split("/")
        date_seg = next((s for s in segs if s.startswith("date=")), None)
        if date_seg:
            d = date_seg.split("=", 1)[1]
            parts_per_date[d] = parts_per_date.get(d, 0) + 1

    report["storage"]["total_bytes"] = int(sum(sizes))
    report["storage"]["small_files_count"] = int(sum(1 for s in sizes if s < small_file_threshold_bytes))
    report["storage"]["parts_per_date_max"] = int(max(parts_per_date.values()) if parts_per_date else 0)
    report["storage"]["parts_per_date_top"] = sorted(parts_per_date.items(), key=lambda x: x[1], reverse=True)[:5]

    # Watermark
    wm = svc.get_watermark(venue, symbol, interval)
    report["watermark"]["duckdb_last_open_time"] = wm

    # Compute time-window dataframe (open_time only) with pushdown
    lf = _scan_ot(files).filter((pl.col("open_time") >= start_ms) & (pl.col("open_time") <= end_ms))

    # Basic counts, min/max
    agg = lf.select(
        pl.len().alias("rows"),
        pl.col("open_time").min().alias("min_open_time"),
        pl.col("open_time").max().alias("max_open_time"),
        pl.col("open_time").n_unique().alias("unique_open_time"),
    ).collect(streaming=True).to_dicts()[0]

    rows = int(agg["rows"])
    urows = int(agg["unique_open_time"])
    report["stats"].update({
        "rows_in_window": rows,
        "unique_open_time_in_window": urows,
        "min_open_time": int(agg["min_open_time"]) if agg["min_open_time"] is not None else None,
        "max_open_time": int(agg["max_open_time"]) if agg["max_open_time"] is not None else None,
    })

    if rows == 0:
        report["checks"]["has_rows_in_window"] = False
        report["recommendations"].append("No rows in requested window; verify start_ms/end_ms or run backfill for that range.")
        return report

    # Duplicates within window
    dup_count = rows - urows
    report["checks"]["duplicates_in_window"] = int(dup_count)
    report["checks"]["duplicate_ratio"] = float(dup_count / rows) if rows > 0 else 0.0

    # Extract sorted unique times to test monotonicity and gaps
    # For large windows you may want to cap; here we do full window to be exact.
    times = (
        lf.select(pl.col("open_time"))
          .unique()
          .sort("open_time")
          .collect(streaming=True)["open_time"]
          .to_list()
    )

    step = INTERVAL_MS.get(interval)
    if step is None:
        report["checks"]["gap_check"] = "skip_unknown_interval"
    else:
        gaps = _find_gaps_from_sorted_times(times, step=step, max_report=max_gap_report)
        report["checks"]["gap_count_reported"] = len(gaps)
        report["checks"]["gaps_sample"] = gaps

        # Fast “is_sorted” check is redundant since we sorted; but we can still validate step regularity:
        # expected number of bars if fully continuous across [min,max] (inclusive count)
        min_ot = times[0]
        max_ot = times[-1]
        expected_if_continuous = ((max_ot - min_ot) // step) + 1
        report["stats"]["expected_if_continuous_between_min_max"] = int(expected_if_continuous)
        report["stats"]["missing_bars_between_min_max"] = int(expected_if_continuous - len(times))

    # Daily completeness check (rows per UTC day)
    daily = (
        _scan_daily_counts(files)
        .filter(
            (pl.col("date") >= pl.from_epoch(pl.lit(start_ms), time_unit="ms").dt.date()) &
            (pl.col("date") <= pl.from_epoch(pl.lit(end_ms), time_unit="ms").dt.date())
        )
        .collect(streaming=True)
    )

    report["stats"]["days_covered_in_window"] = int(daily.height)
    report["stats"]["daily_rows_preview"] = daily.head(10).to_dicts()

    expected_day = _expected_rows_per_day(interval)
    report["stats"]["expected_rows_per_day"] = expected_day

    if expected_day is not None and daily.height > 0:
        # Mark days with low row count
        threshold = int(expected_day * severity_row_threshold)
        bad_days = daily.filter(pl.col("rows") < threshold)
        report["checks"]["bad_days_count"] = int(bad_days.height)
        report["checks"]["bad_days_sample"] = bad_days.head(10).to_dicts()

    # Watermark consistency check with local max (overall, not just window)
    local_max = (
        _scan_ot(files)
        .select(pl.col("open_time").max().alias("m"))
        .collect(streaming=True)["m"][0]
    )
    local_max = int(local_max) if local_max is not None else None
    report["watermark"]["local_max_open_time"] = local_max
    if wm is None:
        report["checks"]["watermark_status"] = "missing_watermark"
        report["recommendations"].append("Watermark missing. Consider rebuild_watermark_from_parquet(...) then sync().")
    else:
        if local_max is not None and abs(local_max - wm) > step if step else 0:
            report["checks"]["watermark_status"] = "mismatch"
            report["recommendations"].append("DuckDB watermark differs from local max open_time. Consider rebuilding watermark from parquet.")
        else:
            report["checks"]["watermark_status"] = "ok"

    # Storage recommendations
    if report["storage"]["small_files_count"] >= warn_small_files_count:
        report["recommendations"].append("Many small parquet files detected; run compact_date_partition on recent dates to reduce file count.")
    if report["storage"]["parts_per_date_max"] >= warn_parts_per_day:
        worst = report["storage"]["parts_per_date_top"][0] if report["storage"]["parts_per_date_top"] else None
        report["recommendations"].append(f"High parts-per-day detected (max={report['storage']['parts_per_date_max']}, worst={worst}). Consider compacting that date.")

    # Gap recommendation
    if step is not None and report["checks"].get("gap_count_reported", 0) > 0:
        report["recommendations"].append("Gaps detected. Run ensure_gaps(venue,symbol,interval,start_ms,end_ms).")

    return report


def print_health_report(r: Dict[str, Any]) -> None:
    tgt = r["target"]
    print(f"HealthCheck: {tgt['venue']} {tgt['symbol']} {tgt['interval']} [{tgt['start_ms']}, {tgt['end_ms']}]")
    print("files:", r.get("storage", {}).get("parquet_files"))
    print("rows_in_window:", r.get("stats", {}).get("rows_in_window"),
          "unique:", r.get("stats", {}).get("unique_open_time_in_window"),
          "dups:", r.get("checks", {}).get("duplicates_in_window"))
    print("min/max open_time:", r.get("stats", {}).get("min_open_time"), r.get("stats", {}).get("max_open_time"))
    print("watermark:", r.get("watermark", {}).get("duckdb_last_open_time"),
          "local_max:", r.get("watermark", {}).get("local_max_open_time"),
          "status:", r.get("checks", {}).get("watermark_status"))
    if "gap_count_reported" in r.get("checks", {}):
        print("gaps_reported:", r["checks"]["gap_count_reported"])
        if r["checks"]["gap_count_reported"] > 0:
            print("gaps_sample:", r["checks"]["gaps_sample"][:5])
    if r.get("checks", {}).get("bad_days_count", 0) > 0:
        print("bad_days_count:", r["checks"]["bad_days_count"])
        print("bad_days_sample:", r["checks"]["bad_days_sample"][:5])
    if r.get("recommendations"):
        print("recommendations:")
        for x in r["recommendations"]:
            print(" -", x)


def utc_ms(dt_utc: datetime) -> int:
    return int(dt_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)

if __name__ == "__main__":
    # Example usage (adjust times)
    svc = BacktestBinanceData(data_root="data/binance", state_db="data/binance_state.duckdb")
    now = now_ms()
    end = (now // 60_000) * 60_000 - 60_000
    start = utc_ms(datetime(2017, 1, 1))

    rep = check_data_health(
        svc,
        venue="binance_um",
        symbol="BTCUSDT",
        interval="1m",
        start_ms=start,
        end_ms=end,
    )
    print_health_report(rep)


