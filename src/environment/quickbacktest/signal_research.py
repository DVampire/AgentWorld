from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, List, Dict, Literal, Optional, Tuple
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np

load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environment.quickbacktest.run import signal_to_dataframe
from libs.BinanceDatabase.src.core.time_utils import utc_ms


# ============================================================
# helpers
# ============================================================

def _validate_horizon(h: int) -> None:
    if h is None or h <= 0:
        raise ValueError(f"horizon must be >= 1, got {h}")


def _get_price_series(
    combo_data: pd.DataFrame,
    rolling_window: int,
    use_smoothing: bool,
) -> pd.Series:
    """
    Keep your original smoothing semantics:
    - if use_smoothing: price = rolling mean(close, rolling_window)
    - else: price = close
    """
    close = combo_data["close"]
    if use_smoothing:
        return close.rolling(window=rolling_window, min_periods=rolling_window).mean()
    return close


def _lag_factor_cols_1bar(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """
    Lag factors by 1 bar to avoid same-bar contamination.
    Return is NOT lagged.
    """
    out = df.copy()
    for c in factor_cols:
        if c in out.columns:
            out[c] = out[c].shift(1)
    return out


def _infer_bar_seconds(idx: pd.DatetimeIndex) -> int:
    """
    Infer bar size in seconds from index.
    (No manual override; fully automatic.)
    """
    if len(idx) < 3:
        return 60
    diffs = idx.to_series().diff().dropna().dt.total_seconds().to_numpy()
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return 60
    return int(np.median(diffs))


def _prepare_eval_frame(
    combo_data: pd.DataFrame,
    price: pd.Series,
    horizon: int,
    rolling_window: int,
    *,
    auto_daily_threshold: int = 1400,
    lag_factors_by_1: bool = True,
    signals_already_causal: bool = False,   # ✅ MANUAL: set True if signal already shifted/causal (e.g., daily shift(1)+ffill)
) -> Tuple[pd.DataFrame, List[str], str, bool, bool, int]:
    """
    Manual control version:
    - auto daily switch if horizon > auto_daily_threshold
    - extra lag is controlled by (lag_factors_by_1 and not signals_already_causal)

    Returns:
      df_eval, signal_cols, ret_col, use_daily, apply_extra_lag, h_eff
    """
    signal_cols = [c for c in combo_data.columns if c.startswith("signal")]
    signal_cols = [c for c in signal_cols if c in combo_data.columns]

    bar_seconds = _infer_bar_seconds(combo_data.index)
    bars_per_day = max(1, int(round(86400 / bar_seconds)))

    use_daily = horizon > auto_daily_threshold
    apply_extra_lag = bool(lag_factors_by_1) and (not bool(signals_already_causal))

    if use_daily:
        h_days = int(round(horizon / bars_per_day))
        h_days = max(1, h_days)

        df_d = pd.DataFrame(index=combo_data.index)
        df_d["price"] = price
        for c in signal_cols:
            df_d[c] = combo_data[c]

        df_d = df_d.resample("1D").last()

        ret_col = f"ret_daily_rm{rolling_window}_fwd{h_days}d"
        df_d[ret_col] = df_d["price"].pct_change(periods=h_days).shift(-h_days)

        # Manual extra lag: 1 day
        if apply_extra_lag and signal_cols:
            df_d[signal_cols] = df_d[signal_cols].shift(1)

        return df_d, signal_cols, ret_col, True, apply_extra_lag, h_days

    # bar mode
    df_b = combo_data.copy()
    df_b["price"] = price

    ret_col = f"ret_rm{rolling_window}_fwd{horizon}"
    df_b[ret_col] = df_b["price"].pct_change(periods=horizon).shift(-horizon)

    # Manual extra lag: 1 bar
    if apply_extra_lag and signal_cols:
        df_b = _lag_factor_cols_1bar(df_b, signal_cols)

    return df_b, signal_cols, ret_col, False, apply_extra_lag, horizon


# ============================================================
# API: correlation matrix (Pearson)
# ============================================================

def get_pearson_correlation(
    data_dir: str = None,
    watermark_dir: str = None,
    venue: str = None,
    symbol: str = None,
    start: datetime = None,
    end: datetime = None,
    signal_module: str = "signal_template",
    base_dir: str = None,
    horizon: int = 0,
    rolling_window: int = 1,
    use_smoothing: bool = False,
    lag_factors_by_1: bool = True,
    img_name: str = "pearson_correlation_matrix",
    auto_daily_threshold: int = 1400,
    signals_already_causal: bool = False,   # ✅ MANUAL knob
) -> Any:
    _validate_horizon(horizon)

    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023, 1, 1))

    combo_data = signal_to_dataframe(
        data_dir, watermark_dir, venue, symbol, start_ms, end_ms, signal_module, base_dir
    ).sort_index()

    price = _get_price_series(combo_data, rolling_window=rolling_window, use_smoothing=use_smoothing)

    df_eval, signal_cols, ret_col, use_daily, apply_extra_lag, h_eff = _prepare_eval_frame(
        combo_data=combo_data,
        price=price,
        horizon=horizon,
        rolling_window=rolling_window,
        auto_daily_threshold=auto_daily_threshold,
        lag_factors_by_1=lag_factors_by_1,
        signals_already_causal=signals_already_causal,
    )

    corr_cols = signal_cols + [ret_col]
    correlation_matrix = df_eval[corr_cols].dropna().corr(method="pearson")

    # if base_dir:
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     im = ax.imshow(correlation_matrix.values, cmap="coolwarm", aspect="auto")
    #     ax.set_xticks(range(len(correlation_matrix.columns)))
    #     ax.set_yticks(range(len(correlation_matrix.index)))
    #     ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    #     ax.set_yticklabels(correlation_matrix.index)
    #     plt.colorbar(im)
    #     mode = "daily" if use_daily else "bar"
    #     lag_tag = "extra_lag" if apply_extra_lag else "no_extra_lag"
    #     plt.title(f"Pearson Correlation - {symbol} ({mode}, h={h_eff}, {lag_tag})")
    #     plt.tight_layout()
    #     out_dir = Path(base_dir) / "images"
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     plt.savefig(out_dir / f"{img_name}.png", dpi=150)
    #     plt.close(fig)

    return {"correlation_matrix": correlation_matrix.to_dict(), "correlation_matrix_img": img_name if base_dir else None}


# ============================================================
# API: correlation matrix (Spearman)
# ============================================================

def get_spearman_correlation(
    data_dir: str = None,
    watermark_dir: str = None,
    venue: str = None,
    symbol: str = None,
    start: datetime = None,
    end: datetime = None,
    signal_module: str = "signal_template",
    base_dir: str = None,
    horizon: int = 0,
    rolling_window: int = 1,
    use_smoothing: bool = False,
    lag_factors_by_1: bool = True,
    img_name: str = "spearman_correlation_matrix",
    auto_daily_threshold: int = 1400,
    signals_already_causal: bool = False,   # ✅ MANUAL knob
) -> Any:
    _validate_horizon(horizon)

    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023, 1, 1))

    combo_data = signal_to_dataframe(
        data_dir, watermark_dir, venue, symbol, start_ms, end_ms, signal_module, base_dir
    ).sort_index()

    price = _get_price_series(combo_data, rolling_window=rolling_window, use_smoothing=use_smoothing)

    df_eval, signal_cols, ret_col, use_daily, apply_extra_lag, h_eff = _prepare_eval_frame(
        combo_data=combo_data,
        price=price,
        horizon=horizon,
        rolling_window=rolling_window,
        auto_daily_threshold=auto_daily_threshold,
        lag_factors_by_1=lag_factors_by_1,
        signals_already_causal=signals_already_causal,
    )

    corr_cols = signal_cols + [ret_col]
    correlation_matrix = df_eval[corr_cols].dropna().corr(method="spearman")

    # if base_dir:
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     im = ax.imshow(correlation_matrix.values, cmap="coolwarm", aspect="auto")
    #     ax.set_xticks(range(len(correlation_matrix.columns)))
    #     ax.set_yticks(range(len(correlation_matrix.index)))
    #     ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    #     ax.set_yticklabels(correlation_matrix.index)
    #     plt.colorbar(im)
    #     mode = "daily" if use_daily else "bar"
    #     lag_tag = "extra_lag" if apply_extra_lag else "no_extra_lag"
    #     plt.title(f"Spearman Correlation - {symbol} ({mode}, h={h_eff}, {lag_tag})")
    #     plt.tight_layout()
    #     out_dir = Path(base_dir) / "images"
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     plt.savefig(out_dir / f"{img_name}.png", dpi=150)
    #     plt.close(fig)

    return {"correlation_matrix": correlation_matrix.to_dict(), "correlation_matrix_img": img_name if base_dir else None}


# ============================================================
# API: IC curve over horizons (Spearman)
# ============================================================

def get_ic_curve(
    data_dir: str = None,
    watermark_dir: str = None,
    venue: str = None,
    symbol: str = None,
    start: datetime = None,
    end: datetime = None,
    signal_module: str = "signal_template",
    base_dir: str = None,
    rolling_window: int = 1,
    horizons: list = [1, 3, 5, 10, 20],
    use_smoothing: bool = False,
    lag_factors_by_1: bool = True,
    img_name: str = "ic_curve",
    auto_daily_threshold: int = 1400,
    signals_already_causal: bool = False,   # ✅ MANUAL knob
) -> Any:
    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023, 1, 1))

    combo_data = signal_to_dataframe(
        data_dir, watermark_dir, venue, symbol, start_ms, end_ms, signal_module, base_dir
    ).sort_index()

    price = _get_price_series(combo_data, rolling_window=rolling_window, use_smoothing=use_smoothing)

    signal_cols = [c for c in combo_data.columns if c.startswith("signal")]
    signal_cols = [c for c in signal_cols if c in combo_data.columns]

    ic_rows: Dict[int, Dict[str, float]] = {}

    for h in horizons:
        _validate_horizon(h)

        df_eval, sig_cols, ret_col, use_daily, apply_extra_lag, h_eff = _prepare_eval_frame(
            combo_data=combo_data,
            price=price,
            horizon=h,
            rolling_window=rolling_window,
            auto_daily_threshold=auto_daily_threshold,
            lag_factors_by_1=lag_factors_by_1,
            signals_already_causal=signals_already_causal,
        )

        tmp = df_eval[sig_cols + [ret_col]].dropna()
        if tmp.empty:
            ic_rows[h] = {f: float("nan") for f in sig_cols}
            continue

        ic_s = tmp[sig_cols].corrwith(tmp[ret_col], method="spearman")
        ic_rows[h] = ic_s.to_dict()

    ic_df = pd.DataFrame.from_dict(ic_rows, orient="index").sort_index()
    ic_df.index.name = "horizon"

    fig, ax = plt.subplots(figsize=(10, 6))
    ic_df.plot(ax=ax, marker="o")
    ax.set_title(
        f"IC Curve (Spearman) - {symbol} - {start} to {end}\n"
        f"rm={rolling_window} | smoothing={use_smoothing} | lag1={lag_factors_by_1} | "
        f"auto_daily>{auto_daily_threshold} | signals_already_causal={signals_already_causal}"
    )
    ax.set_xlabel("Horizon (bars)")
    ax.set_ylabel("Spearman IC")
    ax.legend(title="Signal", loc="best")
    fig.tight_layout()

    if base_dir:
        out_dir = Path(base_dir) / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{img_name}.png", dpi=150)
    plt.close(fig)

    return {"ic_curve": f"images/{img_name}.png" if base_dir else None, "ic_table": ic_df}


# ============================================================
# API: bucket test
# ============================================================

def get_bucket_result(
    data_dir: str = None,
    watermark_dir: str = None,
    venue: str = None,
    symbol: str = None,
    start: datetime = None,
    end: datetime = None,
    signal_module: str = "signal_template",
    base_dir: str = None,
    horizon: int = 1,
    rolling_window: int = 1,
    use_smoothing: bool = False,
    lag_factors_by_1: bool = True,
    img_name: str = "bucket_plot",
    auto_daily_threshold: int = 1400,
    signals_already_causal: bool = False,   # ✅ MANUAL knob
) -> Any:
    _validate_horizon(horizon)

    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023, 1, 1))

    combo_data = signal_to_dataframe(
        data_dir, watermark_dir, venue, symbol, start_ms, end_ms, signal_module, base_dir
    ).sort_index()

    price = _get_price_series(combo_data, rolling_window=rolling_window, use_smoothing=use_smoothing)

    df_eval, signal_cols, ret_col, use_daily, apply_extra_lag, h_eff = _prepare_eval_frame(
        combo_data=combo_data,
        price=price,
        horizon=horizon,
        rolling_window=rolling_window,
        auto_daily_threshold=auto_daily_threshold,
        lag_factors_by_1=lag_factors_by_1,
        signals_already_causal=signals_already_causal,
    )

    df_eval = df_eval.dropna(subset=signal_cols + [ret_col])

    bucket_result: Dict[str, Dict[int, float]] = {}
    bucket_plot_df = pd.DataFrame()

    for factor in signal_cols:
        if df_eval[factor].nunique(dropna=True) < 5:
            bucket_result[factor] = {}
            continue

        ranked = df_eval[factor].rank(method="first")
        bucket = pd.qcut(ranked, q=5, labels=False, duplicates="drop")

        bucket_means = df_eval.groupby(bucket)[ret_col].mean()
        bucket_result[factor] = bucket_means.to_dict()
        bucket_plot_df[factor] = bucket_means

    fig, ax = plt.subplots(figsize=(10, 6))
    bucket_plot_df.plot(kind="bar", ax=ax)
    mode = "daily" if use_daily else "bar"
    lag_tag = "extra_lag" if apply_extra_lag else "no_extra_lag"
    ax.set_title(
        f"Bucket Mean Return - {venue or ''} {symbol or ''}\n"
        f"{start if start else '2022-01-01'} to {end if end else '2023-01-01'}\n"
        f"| mode={mode}, h={h_eff}, rm={rolling_window}, smoothing={use_smoothing}, {lag_tag} | "
        f"signals_already_causal={signals_already_causal}"
    )
    ax.set_xlabel("Bucket (0=lowest signal)")
    ax.set_ylabel("Mean Forward Return")
    ax.axhline(0.0, linewidth=1)
    ax.legend(title="Signal", loc="best")
    fig.tight_layout()

    if base_dir:
        out_dir = Path(base_dir) / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{img_name}.png", dpi=150)
    plt.close(fig)

    return {
        "bucket_result": bucket_result,
        "bucket_plot": f"images/{img_name}.png",
        "bucket_table": bucket_plot_df,
    }


# ============================================================
# API: rolling IC curve (Pearson rolling corr)
# ============================================================

def get_rolling_ic_curve(
    data_dir: str = None,
    watermark_dir: str = None,
    venue: str = None,
    symbol: str = None,
    start: datetime = None,
    end: datetime = None,
    signal_module: str = "signal_template",
    base_dir: str = None,
    rolling_window: int = 20,
    horizon: int = 5,
    ic_window: int = 10,
    use_smoothing: bool = False,
    lag_factors_by_1: bool = True,
    factor: Literal["signal_1", "signal_2", "signal_3"] = "signal_1",
    img_name: str = "rolling_ic_signal",
    auto_daily_threshold: int = 1400,
    signals_already_causal: bool = False,   # ✅ MANUAL knob
) -> Any:
    _validate_horizon(horizon)

    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023, 1, 1))

    combo_data = signal_to_dataframe(
        data_dir, watermark_dir, venue, symbol, start_ms, end_ms, signal_module, base_dir
    )
    if combo_data is None or combo_data.empty:
        raise ValueError("combo_data is empty")

    combo_data = combo_data.sort_index()

    price = _get_price_series(combo_data, rolling_window=rolling_window, use_smoothing=use_smoothing)

    df_eval, signal_cols, ret_col, use_daily, apply_extra_lag, h_eff = _prepare_eval_frame(
        combo_data=combo_data,
        price=price,
        horizon=horizon,
        rolling_window=1,  # keep ret definition independent from price smoothing window semantics
        auto_daily_threshold=auto_daily_threshold,
        lag_factors_by_1=lag_factors_by_1,
        signals_already_causal=signals_already_causal,
    )
    if use_daily:
        ic_window_eff = max(5, int(round(ic_window / 1440)))  # convert to days, ensure at least 5 for stability
    else:
        ic_window_eff = ic_window

    factors = [factor]
    factors = [c for c in factors if c in df_eval.columns and df_eval[c].notna().any()]
    if not factors:
        raise ValueError("No valid factors found (factor not present or all NaN).")

    aligned = df_eval[factors + [ret_col]].dropna()
    if aligned.empty:
        raise ValueError("No overlapping non-NaN samples for factor and forward return.")

    rolling_ic_df = pd.DataFrame(index=aligned.index)
    y = aligned[ret_col]

    for f in factors:
        rolling_ic_df[f] = aligned[f].rolling(window=ic_window_eff, min_periods=ic_window_eff).corr(y)

    fig, ax = plt.subplots(figsize=(12, 6))
    rolling_ic_df.plot(ax=ax)
    ax.axhline(0.0, linewidth=1)

    mode = "daily" if use_daily else "bar"
    lag_tag = "extra_lag" if apply_extra_lag else "no_extra_lag"

    ax.set_title(
        f"Rolling IC (Pearson) - {venue or ''} {symbol or ''}\n"
        f"{start if start else '2022-01-01'} to {end if end else '2023-01-01'}\n"
        f"| mode={mode}, h={h_eff}, ic_w={ic_window} | "
        f"price_rw={rolling_window}, smoothing={use_smoothing}, {lag_tag} | "
        f"signals_already_causal={signals_already_causal}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling IC")
    ax.legend(title="Signal", loc="best")
    fig.tight_layout()

    if base_dir:
        out_dir = Path(base_dir) / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{img_name}.png", dpi=150)
    plt.close(fig)

    return {"rolling_ic_curve": [f"images/{img_name}.png"]}


# ============================================================
# Wrapper: run all
# ============================================================

def _signal_analyzer(
    data_dir: str = None,
    watermark_dir: str = None,
    venue: str = None,
    symbol: str = None,
    start: datetime = None,
    end: datetime = None,
    signal_module: str = "signal_template",
    base_dir: str = None,
    # NEW manual controls
    auto_daily_threshold: int = 1400,
    signals_already_causal: bool = True,   # your current signals (daily resample + shift(1) + ffill) => True
) -> Any:
    pearson_result = get_pearson_correlation(
        data_dir, watermark_dir, venue, symbol, start, end, signal_module, base_dir,
        img_name=f"{signal_module}_pearson_correlation_matrix",
        horizon=1440,
        auto_daily_threshold=auto_daily_threshold,
        signals_already_causal=signals_already_causal,
    )
    spearman_result = get_spearman_correlation(
        data_dir, watermark_dir, venue, symbol, start, end, signal_module, base_dir,
        img_name=f"{signal_module}_spearman_correlation_matrix",
        horizon=1440,
        auto_daily_threshold=auto_daily_threshold,
        signals_already_causal=signals_already_causal,
    )
    ic_curve_result = get_ic_curve(
        data_dir, watermark_dir, venue, symbol, start, end, signal_module, base_dir,
        img_name=f"{signal_module}_ic_curve",
        rolling_window=1,
        horizons=[1440, 3*1440, 5*1440, 10*1440, 20*1440],
        use_smoothing=False,
        lag_factors_by_1=True,
        auto_daily_threshold=auto_daily_threshold,
        signals_already_causal=signals_already_causal,
    )
    bucket_result = get_bucket_result(
        data_dir, watermark_dir, venue, symbol, start, end, signal_module, base_dir,
        img_name=f"{signal_module}_bucket_plot",
        horizon=1440,
        rolling_window=1,
        use_smoothing=False,
        lag_factors_by_1=True,
        auto_daily_threshold=auto_daily_threshold,
        signals_already_causal=signals_already_causal,
    )
    # rolling_ic_signal_1 = get_rolling_ic_curve(
    #     data_dir, watermark_dir, venue, symbol, start, end, signal_module, base_dir,
    #     img_name=f"{signal_module}_rolling_ic_signal_1",
    #     rolling_window=20,
    #     horizon=1440,
    #     ic_window=4000,
    #     use_smoothing=False,
    #     lag_factors_by_1=True,
    #     factor="signal_1",
    #     auto_daily_threshold=auto_daily_threshold,
    #     signals_already_causal=signals_already_causal,
    # )
    # rolling_ic_signal_2 = get_rolling_ic_curve(
    #     data_dir, watermark_dir, venue, symbol, start, end, signal_module, base_dir,
    #     img_name=f"{signal_module}_rolling_ic_signal_2",
    #     rolling_window=20,
    #     horizon=1440,
    #     ic_window=4000,
    #     use_smoothing=False,
    #     lag_factors_by_1=True,
    #     factor="signal_2",
    #     auto_daily_threshold=auto_daily_threshold,
    #     signals_already_causal=signals_already_causal,
    # )
    # rolling_ic_signal_3 = get_rolling_ic_curve(
    #     data_dir, watermark_dir, venue, symbol, start, end, signal_module, base_dir,
    #     img_name=f"{signal_module}_rolling_ic_signal_3",
    #     rolling_window=20,
    #     horizon=1440,
    #     ic_window=4000,
    #     use_smoothing=False,
    #     lag_factors_by_1=True,
    #     factor="signal_3",
    #     auto_daily_threshold=auto_daily_threshold,
    #     signals_already_causal=signals_already_causal,
    # )

    return {
        "pearson_correlation": pearson_result,
        "spearman_correlation": spearman_result,
        "ic_curve": ic_curve_result,
        "bucket_result": bucket_result,
        # "rolling_ic_signal_1": rolling_ic_signal_1,
        # "rolling_ic_signal_2": rolling_ic_signal_2,
        # "rolling_ic_signal_3": rolling_ic_signal_3,
    }