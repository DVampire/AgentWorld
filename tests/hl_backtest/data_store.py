import os
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from hyperliquid.utils import constants
import sqlite3  # NEW: for reading local SQLite .db


def ms_now() -> int:
    return int(time.time() * 1000)


def interval_to_ms(interval: str) -> int:
    """'1m','5m','1h','1d' -> 毫秒"""
    if interval.endswith("m"):
        return int(interval[:-1]) * 60 * 1000
    if interval.endswith("h"):
        return int(interval[:-1]) * 60 * 60 * 1000
    if interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60 * 60 * 1000
    raise ValueError(f"Unknown interval: {interval}")


def _info_url(use_testnet: bool) -> str:
    """
    从 Hyperliquid SDK 拿 REST base url，然后拼 '/info'
    MAINNET_API_URL / TESTNET_API_URL 见官方常量。
    """
    base = constants.TESTNET_API_URL if use_testnet else constants.MAINNET_API_URL
    return base.rstrip("/") + "/info"


def fetch_candles_once(
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    use_testnet: bool = False,
) -> pd.DataFrame:
    """
    调用 /info candleSnapshot 拉一段 K 线。
    注意：官方只能拿最近 5000 根，超出的历史必须自己平时累积存。
    """
    url = _info_url(use_testnet)
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
        },
    }

    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list) or len(data) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # t: open time, T: close time
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df["T"] = pd.to_datetime(df["T"], unit="ms")

    # 转 float
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("t").reset_index(drop=True)
    return df


def ensure_history_csv(
    coin: str,
    interval: str,
    lookback_candles: int,
    data_dir: str = "data",
    use_testnet: bool = False,
    price_col: str = "c",
    local: bool = True,
) -> pd.DataFrame:
    """
    可增量存储的历史数据管理：

    - 第一次运行：拉 lookback_candles 根 K 线（受 5000 限制），保存到 data/{coin}_{interval}.csv
    - 后续运行：读取本地 CSV，根据最后一根 T，向后再拉一段新的 K 线，append 并去重，再保存。
    - 返回最新的 DataFrame（时间升序）。

    ⚠️ 注意：
    - Hyperliquid API 只提供「最近 ~5000 根」的 K 线，如果你很久不更新，中间缺的数据是补不回来的。
      所以建议经常跑一下，把数据接上。
    """
    os.makedirs(data_dir, exist_ok=True)
    path = Path(data_dir) / f"{coin}_{interval}.csv"
    step_ms = interval_to_ms(interval)
    now = ms_now()

    if not path.exists():
        # 第一次：从 now 倒推 lookback_candles * interval（最多 5000）
        n = min(lookback_candles, 5000)
        start = now - n * step_ms
        df = fetch_candles_once(coin, interval, start, now, use_testnet=use_testnet)
        if df.empty:
            raise RuntimeError("No candles fetched from Hyperliquid.")
        # 只保留最后 lookback_candles 根
        df = df.tail(lookback_candles).reset_index(drop=True)
        df.to_csv(path, index=False)
        return df

    # 已有本地 CSV：先读出来
    df_old = pd.read_csv(path, parse_dates=["t", "T"])
    df_old = df_old.sort_values("t").reset_index(drop=True)

    if df_old.empty:
        # 意外空文件，当成第一次
        start = now - lookback_candles * step_ms
        df_new = fetch_candles_once(coin, interval, start, now, use_testnet=use_testnet)
        if df_new.empty:
            raise RuntimeError("No candles fetched from Hyperliquid.")
        df_new = df_new.tail(lookback_candles).reset_index(drop=True)
        df_new.to_csv(path, index=False)
        return df_new

    if local:
        return df_old

    # 以 CSV 里最后一根 K 线的 close time 为起点向后拉
    last_T = df_old["T"].iloc[-1]
    start_ms = int(last_T.timestamp() * 1000) + step_ms
    if start_ms >= now:
        # 已经最新了
        df = df_old.tail(lookback_candles).reset_index(drop=True)
        df.to_csv(path, index=False)
        return df

    df_new = fetch_candles_once(coin, interval, start_ms, now, use_testnet=use_testnet)

    if df_new.empty:
        # API 没有给新数据（比如正好在当前这根 K 线中途），直接返回老的
        df = df_old.tail(lookback_candles).reset_index(drop=True)
        df.to_csv(path, index=False)
        return df

    # 合并 & 去重（按 T）
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all = df_all.drop_duplicates(subset="T").sort_values("t").reset_index(drop=True)

    # 只保留最后 lookback_candles 根，避免 CSV 无限变大
    df = df_all.tail(lookback_candles).reset_index(drop=True)
    df.to_csv(path, index=False)

    return df


def load_klines_from_sqlite(
    db_path: str = "database.db",
    table_name: str = "data_BTC_candle",
) -> pd.DataFrame:
    """
    从本地 SQLite K 线表中读取数据，并转换为统一格式：

    列: t, T, s, i, o, c, h, l, v
    - t: timestamp_utc -> pandas datetime
    - T: timestamp_utc -> 字符串格式 'YYYY-MM-DD HH:MM:SS.sss'
    - s: symbol
    - i: interval
    - o: open
    - c: close
    - h: high
    - l: low
    - v: volume
    """
    conn = sqlite3.connect(db_path)

    # 基础 SQL
    base_sql =f"""
SELECT
    timestamp_utc,     
    timestamp_utc,     
    symbol,            
    interval,          
    open,              
    close,             
    high,           
    low,            
    volume            
FROM {table_name};
"""

    df = pd.read_sql_query(base_sql, conn)
    conn.close()

    # 重命名列
    df.columns = ["t", "T", "s", "i", "o", "c", "h", "l", "v"]

    # t: datetime
    df["t"] = pd.to_datetime(df["t"])

    # T: 转为毫秒精度字符串
    df["T"] = pd.to_datetime(df["T"], utc=False)
    df["T"] = df["T"].dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]

    # 类型转换（价格/量转 float）
    for col in ["o", "c", "h", "l", "v"]:
        df[col] = df[col].astype(float)

    return df
