from __future__ import annotations

import os
import time
import glob
from typing import List, Optional, Literal

import polars as pl
import shutil

from .constants import KLINE_COLS

class ParquetLake:
    def __init__(self, data_root: str):
        self.data_root = data_root
        os.makedirs(self.data_root, exist_ok=True)

    def to_polars(self, klines: List[list], venue: str, symbol: str, interval: str) -> pl.DataFrame:
        df = pl.DataFrame(klines, schema=KLINE_COLS, orient="row")

        df = df.with_columns(
            pl.lit(venue).alias("venue"),
            pl.lit(symbol).alias("symbol"),
            pl.lit(interval).alias("interval"),
            pl.col("open_time").cast(pl.Int64),
            pl.col("close_time").cast(pl.Int64),
            pl.col("num_trades").cast(pl.Int64),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64),
            pl.col("taker_buy_base_vol").cast(pl.Float64),
            pl.col("taker_buy_quote_vol").cast(pl.Float64),
        ).drop("ignore")

        df = df.with_columns(
            pl.from_epoch("open_time", time_unit="ms").dt.date().cast(pl.Utf8).alias("date")
        )

        cols = [
            "venue","symbol","interval","open_time","open","high","low","close","volume",
            "close_time","quote_volume","num_trades","taker_buy_base_vol","taker_buy_quote_vol","date"
        ]
        return df.select(cols)

    def write_partitioned(self, df: pl.DataFrame) -> None:
        parts = df.partition_by(["venue", "symbol", "interval", "date"], as_dict=True)
        for (v, s, i, d), g in parts.items():
            out_dir = os.path.join(self.data_root, f"venue={v}", f"symbol={s}", f"interval={i}", f"date={d}")
            os.makedirs(out_dir, exist_ok=True)

            g = g.sort("open_time").unique(
                subset=["venue","symbol","interval","open_time"],
                keep="last",
            )
            fn = f"part-{int(time.time() * 1000)}.parquet"
            g.write_parquet(os.path.join(out_dir, fn))

    def query_scan(
            self,
            venue: str,
            symbol: str,
            interval: str,
            start_ms: int,
            end_ms: int,
            *,
            columns: Optional[List[str]] = None,
            as_: Literal["polars", "pandas"] = "polars",
        ):
            base = os.path.join(self.data_root, f"venue={venue}", f"symbol={symbol}", f"interval={interval}")
            empty = pl.DataFrame()

            if not os.path.exists(base):
                return empty if as_ == "polars" else empty.to_pandas()

            # 关键修复：glob 先展开，若为空则直接返回空 DF
            glob_path = os.path.join(base, "**", "*.parquet")
            files = glob.glob(glob_path, recursive=True)
            if not files:
                return empty if as_ == "polars" else empty.to_pandas()

            # ---- 下面才开始 scan ----
            dedup_keys = ["venue", "symbol", "interval", "open_time"]
            lf = pl.scan_parquet(files)  # 直接传 files 列表，避免空 glob

            # 如果传了 columns，补齐去重所需列（修复你之前的 interval 缺失问题）
            if columns:
                select_cols = list(dict.fromkeys(columns + [c for c in dedup_keys if c not in columns]))
                lf = lf.select(select_cols)

            lf = lf.filter((pl.col("open_time") >= int(start_ms)) & (pl.col("open_time") <= int(end_ms)))

            # 去重列存在性防御
            existing = set(lf.collect_schema().names())
            subset = [c for c in dedup_keys if c in existing]
            if subset:
                lf = lf.sort("open_time").unique(subset=dedup_keys, keep="last", maintain_order=True)
            else:
                lf = lf.sort("open_time")

            out = lf.collect(streaming=True)

            # 返回列裁剪
            if columns:
                # 如果用户只想要原 columns，这里再裁回去
                out = out.select([c for c in columns if c in out.columns])

            return out if as_ == "polars" else out.to_pandas()
    

    def _partition_dir(self, venue: str, symbol: str, interval: str, date: str) -> str:
        return os.path.join(
            self.data_root,
            f"venue={venue}",
            f"symbol={symbol}",
            f"interval={interval}",
            f"date={date}",
        )

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
    ) -> dict:
        """
        Compact one date partition:
          - read all part-*.parquet in partition dir
          - dedup by (venue,symbol,interval,open_time), keep last
          - sort by open_time
          - write one or multiple larger parquet files
          - atomically replace old files

        force:
          - False: skip if files count <= 1 (nothing to compact)
          - True: always rewrite (useful if you want to guarantee dedup/sort)
        """
        part_dir = self._partition_dir(venue, symbol, interval, date)
        if not os.path.exists(part_dir):
            return {"date": date, "status": "missing_partition", "written_files": 0, "input_files": 0}

        inputs = sorted(glob.glob(os.path.join(part_dir, "*.parquet")))
        if not inputs:
            return {"date": date, "status": "empty_partition", "written_files": 0, "input_files": 0}

        if (len(inputs) <= 1) and not force:
            return {"date": date, "status": "skip_small", "written_files": 0, "input_files": len(inputs)}

        # temp dir beside partition (same filesystem ensures rename atomic)
        tmp_dir = part_dir + f".__compact_tmp__{int(time.time()*1000)}"
        os.makedirs(tmp_dir, exist_ok=True)

        try:
            # Lazy scan all files in this partition only (fast, bounded)
            lf = pl.scan_parquet(inputs)

            # Ensure required cols exist; if user previously used projections, partition files still have full schema
            dedup_keys = ["venue", "symbol", "interval", "open_time"]
            lf = lf.sort("open_time").unique(subset=dedup_keys, keep="last", maintain_order=True)

            # Collect in streaming mode to reduce peak memory
            df = lf.collect(streaming=True)

            if df.is_empty():
                # write nothing; keep original partition
                shutil.rmtree(tmp_dir, ignore_errors=True)
                return {"date": date, "status": "no_rows_after_compact", "written_files": 0, "input_files": len(inputs)}

            # Chunked write (avoid creating one massive file if huge)
            n = df.height
            written_files = 0
            start = 0
            while start < n:
                end = min(n, start + target_rows_per_file)
                chunk = df.slice(start, end - start)
                out_fn = f"compact-{written_files:04d}.parquet"
                chunk.write_parquet(os.path.join(tmp_dir, out_fn))
                written_files += 1
                start = end

            # Atomic replace:
            # 1) move old files to backup dir (optional but safer), then
            # 2) move compacted files into partition dir, then
            # 3) delete backup if requested
            backup_dir = part_dir + f".__backup__{int(time.time()*1000)}"
            os.makedirs(backup_dir, exist_ok=True)

            # move old inputs to backup
            for f in inputs:
                shutil.move(f, os.path.join(backup_dir, os.path.basename(f)))

            # move new compact files into partition dir
            for f in sorted(glob.glob(os.path.join(tmp_dir, "*.parquet"))):
                shutil.move(f, os.path.join(part_dir, os.path.basename(f)))

            # cleanup
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if delete_inputs:
                shutil.rmtree(backup_dir, ignore_errors=True)
            else:
                # keep backup_dir as archive
                pass

            return {
                "date": date,
                "status": "compacted",
                "input_files": len(inputs),
                "written_files": written_files,
                "rows": int(n),
                "partition_dir": part_dir,
            }

        except Exception as e:
            # rollback best-effort: keep original partition intact if possible
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
    
    def max_open_time(self, venue: str, symbol: str, interval: str) -> int | None:
        base = os.path.join(self.data_root, f"venue={venue}", f"symbol={symbol}", f"interval={interval}")
        if not os.path.exists(base):
            return None

        files = glob.glob(os.path.join(base, "**", "*.parquet"), recursive=True)
        if not files:
            return None

        # 只读 open_time 列，成本很低
        lf = pl.scan_parquet(files).select(pl.col("open_time").max().alias("max_ot"))
        m = lf.collect(streaming=True)["max_ot"][0]
        return int(m) if m is not None else None


    



REQUIRED_COLS = [
    "open_time","open","high","low","close","volume",
    "close_time","quote_volume","num_trades",
    "taker_buy_base_vol","taker_buy_quote_vol",
]

def _normalize_import_df(
    df,
    venue: str,
    symbol: str,
    interval: str,
) -> pl.DataFrame:
    """
    Normalize pandas / polars df into canonical polars schema.
    """
    if not isinstance(df, pl.DataFrame):
        df = pl.from_pandas(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"import df missing columns: {missing}")

    out = (
        df
        .with_columns(
            pl.lit(venue).alias("venue"),
            pl.lit(symbol).alias("symbol"),
            pl.lit(interval).alias("interval"),
            pl.col("open_time").cast(pl.Int64),
            pl.col("close_time").cast(pl.Int64),
            pl.col("num_trades").cast(pl.Int64),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64),
            pl.col("taker_buy_base_vol").cast(pl.Float64),
            pl.col("taker_buy_quote_vol").cast(pl.Float64),
        )
        .with_columns(
            pl.from_epoch("open_time", time_unit="ms")
              .dt.date()
              .cast(pl.Utf8)
              .alias("date")
        )
    )

    cols = [
        "venue","symbol","interval","open_time","open","high","low","close","volume",
        "close_time","quote_volume","num_trades",
        "taker_buy_base_vol","taker_buy_quote_vol","date",
    ]
    return out.select(cols)

