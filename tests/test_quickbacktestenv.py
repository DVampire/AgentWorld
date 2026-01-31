from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environment import QuickBacktestEnvironment
import asyncio

env = QuickBacktestEnvironment(base_dir="workdir/trading_strategy_agent/quickbacktest")

async def setup_environment():
    await env.initialize()
    signal_code = '''

    <code>
from Signals.types import BaseSignal
from typing import Literal, Optional, Tuple
import pandas as pd
import numpy as np
import talib as ta


class AgentSignal(BaseSignal):
    """
    AgentSignal
    ===========

    This class prepares trading inputs for the strategy.
    It does NOT execute trades.
    Avoid look-head bias when generating signals and factors.

    {
        "signal":{
            "name":"CS_ZSCORE_MOM_RET",
            "explain":"Cross-sectional z-scored momentum. For each code, compute close-to-close log return over a lookback window (shifted by 1 to avoid look-ahead), then z-score across codes each trade_time. Higher implies stronger recent performance vs peers."
        },
        "factor1":{
            "name":"CS_ZSCORE_VOL_RET",
            "explain":"Cross-sectional z-scored volatility. For each code, compute rolling standard deviation of daily log returns (shifted by 1). Z-score across codes each trade_time. Higher implies higher recent realized volatility vs peers."
        },
        "factor2":{
            "name":"CS_ZSCORE_LIQ_LOGVOL",
            "explain":"Cross-sectional z-scored liquidity proxy via rolling mean of log(volume) (shifted by 1). Z-score across codes each trade_time. Higher implies more recent trading activity vs peers."
        }
    }

    Outputs consumed by Strategy
    ----------------------------
    - signal   : primary decision input (price / score / indicator)
    - factor1  : auxiliary factor
    - factor2  : auxiliary factor
    """

    # ----------------------------
    # Helpers: detect schema + pivot
    # ----------------------------
    def _infer_schema_and_pivot(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Return wide OHLCV matrices: close_w, open_w, high_w, low_w, volume_w
        index=trade_time, columns=code
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise ValueError("`data` must be a non-empty pd.DataFrame")

        cols = set(data.columns)

        # Case A: already wide (common: columns as MultiIndex like ('close', code))
        if isinstance(data.columns, pd.MultiIndex):
            # Expect first level in {open, high, low, close, volume}
            lvl0 = data.columns.get_level_values(0)
            def pick(field: str) -> pd.DataFrame:
                if field not in set(lvl0):
                    raise ValueError(f"Missing `{field}` in wide MultiIndex columns.")
                out = data.loc[:, field]
                out.index.name = out.index.name or "trade_time"
                return out

            close_w = pick("close")
            open_w = pick("open") if "open" in set(lvl0) else close_w.copy()
            high_w = pick("high") if "high" in set(lvl0) else close_w.copy()
            low_w  = pick("low")  if "low"  in set(lvl0) else close_w.copy()
            vol_w  = pick("volume") if "volume" in set(lvl0) else pd.DataFrame(
                np.nan, index=close_w.index, columns=close_w.columns
            )
            return close_w, open_w, high_w, low_w, vol_w

        # Case B: long format with trade_time, code, ohlcv columns
        required = {"trade_time", "code"}
        if required.issubset(cols):
            # common column names
            # close is required for our implementation
            if "close" not in cols:
                raise ValueError("Long-format data must contain `close` column.")

            def pivot(field: str, fill: float = np.nan) -> pd.DataFrame:
                if field not in cols:
                    # optional fields
                    wide = (
                        data[["trade_time", "code"]]
                        .assign(**{field: fill})
                        .pivot(index="trade_time", columns="code", values=field)
                    )
                else:
                    wide = data.pivot(index="trade_time", columns="code", values=field)
                wide.index = pd.to_datetime(wide.index)
                wide = wide.sort_index()
                return wide

            close_w = pivot("close")
            open_w  = pivot("open") if "open" in cols else close_w.copy()
            high_w  = pivot("high") if "high" in cols else close_w.copy()
            low_w   = pivot("low")  if "low"  in cols else close_w.copy()
            vol_w   = pivot("volume") if "volume" in cols else pd.DataFrame(
                np.nan, index=close_w.index, columns=close_w.columns
            )
            return close_w, open_w, high_w, low_w, vol_w

        raise ValueError(
            "Unrecognized data schema. Provide either:\n"
            "1) long format with columns: trade_time, code, close (and optional open/high/low/volume), or\n"
            "2) wide MultiIndex columns like ('close', code), ('volume', code), etc."
        )

    @staticmethod
    def _cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score across columns at each row (trade_time)."""
        mu = x.mean(axis=1)
        sd = x.std(axis=1, ddof=0).replace(0.0, np.nan)
        return x.sub(mu, axis=0).div(sd, axis=0)

    @staticmethod
    def _safe_shift(df: pd.DataFrame, n: int = 1) -> pd.DataFrame:
        """Shift by n periods to avoid using same-bar information."""
        return df.shift(n)

    # ----------------------------
    # Core computations
    # ----------------------------
    def get_signal(self, **kwargs) -> pd.DataFrame:
        """
        Generate the primary signal: cross-sectional z-scored momentum.
        Expected kwargs:
            - data: pd.DataFrame (required)
            - mom_window: int (default 20)
            - lag: int (default 1)  # ensures no look-ahead
        """
        data = kwargs.get("data", None)
        mom_window = int(kwargs.get("mom_window", 20))
        lag = int(kwargs.get("lag", 1))

        close_w, *_ = self._infer_schema_and_pivot(data)

        # log returns; use past info only (shifted)
        log_close = np.log(close_w.replace(0, np.nan))
        ret_1 = log_close.diff(1)

        # momentum: sum of returns over window (or equivalently log return over window)
        mom = ret_1.rolling(mom_window, min_periods=mom_window).sum()

        # strictly use information available before current bar
        mom = self._safe_shift(mom, lag)

        # cross-sectional z-score each day
        signal = self._cs_zscore(mom)
        signal.index.name = "trade_time"
        return signal

    def get_factors(
        self,
        factor_name: Literal["factor1", "factor2"],
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate factor1 or factor2.
        Expected kwargs:
            - data: pd.DataFrame (required)
            - vol_window: int (default 20)
            - liq_window: int (default 20)
            - lag: int (default 1)
        """
        data = kwargs.get("data", None)
        vol_window = int(kwargs.get("vol_window", 20))
        liq_window = int(kwargs.get("liq_window", 20))
        lag = int(kwargs.get("lag", 1))

        close_w, _, _, _, vol_w = self._infer_schema_and_pivot(data)

        log_close = np.log(close_w.replace(0, np.nan))
        ret_1 = log_close.diff(1)

        if factor_name == "factor1":
            # rolling realized vol (std of returns), shifted
            vol = ret_1.rolling(vol_window, min_periods=vol_window).std(ddof=0)
            vol = self._safe_shift(vol, lag)
            out = self._cs_zscore(vol)

        elif factor_name == "factor2":
            # liquidity proxy: rolling mean of log(volume), shifted
            # (if volume missing -> all NaN; caller should ensure volume exists if needed)
            v = vol_w.replace(0, np.nan)
            logv = np.log(v)
            liq = logv.rolling(liq_window, min_periods=liq_window).mean()
            liq = self._safe_shift(liq, lag)
            out = self._cs_zscore(liq)

        else:
            raise ValueError("factor_name must be 'factor1' or 'factor2'")

        out.index.name = "trade_time"
        return out

    def concat_signal(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Attach signal / factor1 / factor2 back to the original OHLCV data.

        Rules (enforced)
        - kwargs must be read via kwargs.get("x", default)
        - kwargs may be empty
        - Do not drop or reorder original rows
        - Do not interpret trading logic here
        """
        # Parameters
        vwap_window = int(kwargs.get("vwap_window", 20))

        # Compute wide matrices
        signal_w = self.get_signal(data=data, **kwargs)
        factor1_w = self.get_factors("factor1", data=data, **kwargs)
        factor2_w = self.get_factors("factor2", data=data, **kwargs)

        # VWAP is implemented in BaseSignal; assume it returns wide (index=trade_time, columns=code)
        vwaps = self.calculate_rolling_vwap(data=data, window=vwap_window)

        # Convert to long for joins
        signal_long = signal_w.stack(dropna=False).to_frame("signal")
        factor1_long = factor1_w.stack(dropna=False).to_frame("factor1")
        factor2_long = factor2_w.stack(dropna=False).to_frame("factor2")
        vwap_long = vwaps.stack(dropna=False).to_frame("vwap")

        signal_long.index.names = ["trade_time", "code"]
        factor1_long.index.names = ["trade_time", "code"]
        factor2_long.index.names = ["trade_time", "code"]
        vwap_long.index.names = ["trade_time", "code"]

        # Join to original without dropping/reordering:
        # - If original is long: preserve exact row order
        # - If original is wide: cannot preserve row-level order, so we create a long version in sorted order
        if {"trade_time", "code"}.issubset(set(data.columns)):
            base = data.copy()
            base["_row_id"] = np.arange(len(base))  # preserve order
            base = base.set_index(["trade_time", "code"])

            merged = (
                base.join([signal_long, factor1_long, factor2_long, vwap_long], how="left")
                    .reset_index()
                    .sort_values("_row_id", kind="stable")
                    .drop(columns=["_row_id"])
            )
            return merged

        # Wide input: produce a standard long table (best-effort, stable deterministic ordering)
        # Note: we cannot "not reorder original rows" because there are no original (trade_time, code) rows.
        close_w, *_ = self._infer_schema_and_pivot(data)
        idx = close_w.index
        cols = close_w.columns
        base_long = (
            pd.MultiIndex.from_product([idx, cols], names=["trade_time", "code"])
            .to_frame(index=False)
        )
        merged = (
            base_long.set_index(["trade_time", "code"])
                    .join([signal_long, factor1_long, factor2_long, vwap_long], how="left")
                    .reset_index()
                    .sort_values(["trade_time", "code"])
        )
        return merged </code>
'''
    await env.addSignal(signal_code=signal_code, signal_name="AgentSignal")
    signals = await env.listSignals()
    return signals

asyncio.run(setup_environment())