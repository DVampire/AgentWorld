import time
from datetime import datetime, timezone, timedelta

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

    # -----------------------------
    # A) 导入历史（可选：你有外部 CSV/Vendor 数据时）
    # -----------------------------
    # 假设你已有一个 polars/pandas df（必须包含 open_time...等列）
    # df_hist = pl.read_csv("your_hist.csv")
    # svc.import_klines(df_hist, venue=venue, symbol=symbol, interval="1m", update_watermark=True)

    # -----------------------------
    # B) 定时/手动增量更新（幂等）
    # -----------------------------
    # 建议：start_ms 设为你希望保存的最早历史（或你的历史导入起点）
    start_ms = utc_ms(datetime(2017, 1, 1))
    r = svc.sync(venue, symbol, "1m", start_ms)
    print("sync:", r)

    # -----------------------------
    # C) 查询（给策略/回测）
    # -----------------------------
    # 取最近 N 小时的已收盘分钟数据
    end_closed_open = floor_to_minute_ms(now_ms()) - 60_000
    start_window = end_closed_open - 24 * 60 * 60 * 1000  # last 24h

    df_1m = svc.query(
        venue, symbol, "1m",
        start_window, end_closed_open,
        as_="polars",
        columns=["open_time", "open", "high", "low", "close", "volume"],
    )

    if df_1m.is_empty():
        print("No local data; consider backfill or check symbol/venue.")
        return

    print("queried 1m rows:", df_1m.height)
    print(df_1m.tail(5))

    # -----------------------------
    # D) 需要高周期就本地重采样（推荐做法）
    # -----------------------------
    svc.resample_from_1m(venue, symbol, start_window, end_closed_open, target_interval="1h")
    df_1h = svc.query(
        venue, symbol, "1h",
        start_window, end_closed_open,
        as_="polars",
        columns=["open_time", "open", "high", "low", "close", "volume"],
    )
    print("queried 1h rows:", df_1h.height)
    print(df_1h.tail(5))

    # -----------------------------
    # E) 数据完整性维护：补齐缺口（建议按天/按周跑）
    # -----------------------------
    # 例如：对最近 7 天检查缺口并回填
    start_7d = end_closed_open - 7 * 24 * 60 * 60 * 1000
    gap = svc.ensure_gaps(venue, symbol, "1m", start_7d, end_closed_open)
    print("ensure_gaps:", gap)

    # -----------------------------
    # F) 物理存储维护：compact（小文件合并，建议每日跑）
    # -----------------------------
    # 只 compact 最近两天，避免全量重写
    # 你需要传入 date=YYYY-MM-DD；日期按 UTC 分区
    today_utc = datetime.now(timezone.utc).date()
    dates = [(today_utc - timedelta(days=i)).isoformat() for i in range(2)]
    for d in dates:
        try:
            out = svc.compact_date_partition(
                venue=venue, symbol=symbol, interval="1m", date=d,
                force=False,           # 文件少就跳过
                delete_inputs=True,
            )
            print("compact:", out)
        except Exception as e:
            # 生产建议：打日志，不要让维护任务阻断主流程
            print("compact error:", repr(e))

    # -----------------------------
    # G) 交给回测/因子引擎：示例信号计算
    # -----------------------------
    # 例如用 close 做一个简单的收益率序列
    # 注意：这里只示例，你自己的回测逻辑接入 df_1m 即可
    close = df_1m["close"]
    ret = close.pct_change().fill_null(0.0)
    print("example ret tail:", ret.tail(5))


if __name__ == "__main__":
    main()
