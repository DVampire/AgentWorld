from __future__ import annotations

import polars as pl
from .constants import INTERVAL_MS


def resample_1m_to_target(
    base_1m: pl.DataFrame,
    venue: str,
    symbol: str,
    target_interval: str,
) -> pl.DataFrame:
    rule_map = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
    if target_interval not in rule_map:
        raise ValueError("supports only 5m/15m/1h/4h/1d in this version")

    every = rule_map[target_interval]

    base = base_1m.with_columns(
        pl.from_epoch("open_time", time_unit="ms").alias("ts")
    ).sort("ts")

    out = (
        base.group_by_dynamic("ts", every=every, closed="left", label="left")
        .agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("quote_volume").sum().alias("quote_volume"),
            pl.col("num_trades").sum().cast(pl.Int64).alias("num_trades"),
            pl.col("taker_buy_base_vol").sum().alias("taker_buy_base_vol"),
            pl.col("taker_buy_quote_vol").sum().alias("taker_buy_quote_vol"),
        ])
        .drop_nulls()
    )

    out = out.with_columns(
        pl.lit(venue).alias("venue"),
        pl.lit(symbol).alias("symbol"),
        pl.lit(target_interval).alias("interval"),
        (pl.col("ts").cast(pl.Datetime("ms")).cast(pl.Int64)).alias("open_time"),
    ).with_columns(
        (pl.col("open_time") + pl.lit(INTERVAL_MS[target_interval] - 1)).alias("close_time"),
        pl.col("ts").dt.date().cast(pl.Utf8).alias("date"),
    ).drop("ts")

    cols = [
        "venue","symbol","interval","open_time","open","high","low","close","volume",
        "close_time","quote_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol","date"
    ]
    return out.select(cols)
