from pathlib import Path
import sys
from typing import Any,List,Optional,Dict
from dotenv import load_dotenv
from matplotlib import pyplot as plt
load_dotenv(verbose=True)
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)
from src.environment.quickbacktest.run import signal_to_dataframe
from libs.BinanceDatabase.src.core.time_utils import utc_ms
from datetime import datetime
import pandas as pd
from pathlib import Path




def get_spearman_correlation(data_dir: str = None, watermark_dir: str = None, venue: str = None, symbol: str = None,start: datetime = None,end: datetime = None, signal_module: str = "signal_template",base_dir: str = None,horizon: int = 0,rolling_window: int = 1) -> Any:
    
    
    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023,1,1))
    combo_data = signal_to_dataframe(data_dir,watermark_dir,venue,symbol,start_ms,end_ms,signal_module, base_dir)

    p = combo_data[f"close"].rolling(rolling_window, min_periods=rolling_window).mean()
    combo_data[f"return_rolling={rolling_window}_shift={horizon}"] = p.pct_change(periods=horizon).shift(-horizon)
    factors = ["signal"] + [col for col in combo_data.columns if col.startswith("factor")]+ [f"return_rolling={rolling_window}_shift={horizon}"]
    correlation_matrix = combo_data[factors].corr(method="spearman")
    return correlation_matrix.to_dict()


def get_anything_plot(data_dir: str = None, watermark_dir: str = None, venue: str = None, symbol: str = None,start: datetime = None,end: datetime = None, signal_module: str = "signal_template",base_dir: str = None,plotting_columns: list = None) -> Any:

    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023,1,1))
    combo_data = signal_to_dataframe(data_dir,watermark_dir,venue,symbol,start_ms,end_ms,signal_module, base_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    for col in plotting_columns:
        if col in combo_data.columns:
            ax.plot(combo_data.index, combo_data[col], label=col)
    ax.set_title(f"{symbol} - {plotting_columns} over time- \n {start, end}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()

    plt.savefig(Path(base_dir) / "anything_plot.png")
    plt.close(fig)
    # return {
    #     "anything_plot": str(Path(base_dir) / "anything_plot.png") if base_dir else None
    # }

    return {
        "anything_plot": "anything_plot.png" if base_dir else None
    }


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
    horizons: list = [1,3,5,10,20]
):
    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023,1,1))
    combo_data = signal_to_dataframe(data_dir,watermark_dir,venue,symbol,start_ms,end_ms,signal_module, base_dir)

    # 1) 平滑价格（可选：用 rolling mean 生成更干净的“价格”）
    p = combo_data["close"].rolling(window=rolling_window, min_periods=rolling_window).mean()

    # 2) 需要参与 IC 的列（signal + factor*）
    factors = ["signal"] + [c for c in combo_data.columns if c.startswith("factor")]
    factors = [c for c in factors if c in combo_data.columns]  # 防御：存在才用

    # 3) 计算每个 horizon 的 forward return，并算 Spearman IC
    ic_rows = {}
    for h in horizons:
        ret_col = f"ret_rm{rolling_window}_fwd{h}"
        # forward h-step return: (p_t / p_{t-h} - 1) 对齐到 t（用 shift(-h) 变成 t->t+h）
        combo_data[ret_col] = p.pct_change(periods=h).shift(-h)

        tmp = combo_data[factors + [ret_col]].dropna()
        if tmp.empty:
            ic_rows[h] = {f: float("nan") for f in factors}
            continue

        ic_s = tmp[factors].corrwith(tmp[ret_col], method="spearman")
        ic_rows[h] = ic_s.to_dict()

    # 4) horizon × factor 的 IC 矩阵
    ic_df = pd.DataFrame.from_dict(ic_rows, orient="index").sort_index()
    ic_df.index.name = "horizon"

    # 5) 画图：横轴 horizon，纵轴 IC，每条线一个 factor
    fig, ax = plt.subplots(figsize=(10, 6))
    ic_df.plot(ax=ax, marker="o")
    ax.set_title(f"IC Curve (Spearman) - {symbol} - {start} to {end} - rolling_windows{rolling_window}")
    ax.set_xlabel("Horizon (bars)")
    ax.set_ylabel("Spearman IC")
    ax.legend(title="Factor", loc="best")
    fig.tight_layout()

    out_path = None
    if base_dir:
        out_path = Path(base_dir) / f"ic_curve.png"
        fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # return {
    #     "ic_curve": str(out_path) if out_path else None,
    #     "ic_table": ic_df,  # 方便你后续做筛选/排序/导出
    # }
    return {
        "ic_curve": "ic_curve.png" if base_dir else None,
        "ic_table": ic_df,  # 方便你后续做筛选/排序/导出
    }

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
) -> Any:

    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023, 1, 1))

    combo_data = signal_to_dataframe(
        data_dir, watermark_dir, venue, symbol,
        start_ms, end_ms, signal_module, base_dir
    )

    combo_data = combo_data.sort_index()

    # 1️⃣ 平滑价格 + forward return
    p = combo_data["close"].rolling(
        window=rolling_window,
        min_periods=rolling_window
    ).mean()

    ret_col = f"ret_rm{rolling_window}_fwd{horizon}"
    combo_data[ret_col] = p.pct_change(periods=horizon).shift(-horizon)

    # 2️⃣ 因子列
    factors = ["signal"] + [c for c in combo_data.columns if c.startswith("factor")]
    factors = [c for c in factors if c in combo_data.columns]

    combo_data = combo_data.dropna(subset=factors + [ret_col])

    bucket_result = {}
    bucket_plot_df = pd.DataFrame()

    # 3️⃣ 计算 bucket 均值
    for factor in factors:
        ranked = combo_data[factor].rank(method="first")
        bucket = pd.qcut(ranked, q=5, labels=False)

        bucket_means = combo_data.groupby(bucket)[ret_col].mean()

        bucket_result[factor] = bucket_means.to_dict()
        bucket_plot_df[factor] = bucket_means

    # 4️⃣ Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bucket_plot_df.plot(kind="bar", ax=ax)

    ax.set_title(
        f"Bucket Mean Return - {venue or ''} {symbol or ''}\n"
        f"h={horizon}, rm={rolling_window}"
    )
    ax.set_xlabel("Bucket (0=lowest signal)")
    ax.set_ylabel("Mean Forward Return")
    ax.axhline(0.0, linewidth=1)
    ax.legend(title="Factor", loc="best")
    fig.tight_layout()

    plot_path = None
    if base_dir:
        out_dir = Path(base_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / f"bucket_plot.png"
        fig.savefig(plot_path, dpi=150)

    plt.close(fig)

    # return {
    #     "bucket_result": bucket_result,
    #     "bucket_plot": str(plot_path) if plot_path else None,
    #     "bucket_table": bucket_plot_df,
    # }
    return {
        "bucket_result": bucket_result,
        "bucket_plot": "bucket_plot.png",
        "bucket_table": bucket_plot_df,
    }

def _forward_return(
    price: pd.Series,
    horizon: int,
) -> pd.Series:
    """
    Forward return aligned at time t:
        ret_t = price_{t+h} / price_t - 1
    """
    if horizon <= 0:
        raise ValueError("horizon must be >= 1")
    return price.pct_change(periods=horizon).shift(-horizon)


def _rolling_spearman_ic(
    x: pd.Series,
    y: pd.Series,
    window: int,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Spearman = Pearson(corr(rank(x), rank(y))) within each rolling window.
    This is a standard trick to get rolling Spearman efficiently.
    """
    if min_periods is None:
        min_periods = window

    # rank within the entire series first; then rolling corr of ranks
    # NOTE: For strict Spearman, rank should be computed within each window.
    # This global-rank approximation is common and fast.
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    return xr.rolling(window=window, min_periods=min_periods).corr(yr)


def get_rolling_ic_curve(
    data_dir: str = None,
    watermark_dir: str = None,
    venue: str = None,
    symbol: str = None,
    start: datetime = None,
    end: datetime = None,
    signal_module: str = "signal_template",
    base_dir: str = None,
    rolling_window: int = 20,     # price smoothing window (optional)
    horizon: int = 60,            # forward return horizon (bars)
    ic_window: int = 600,         # rolling IC window (bars)  ✅ default fixed
    method: str = "spearman",     # "pearson" or "spearman"
    use_smoothed_price: bool = False,  # ✅ 默认 False：避免平滑导致伪强相关
    lag_factors_by_1: bool = True,     # ✅ 默认 True：避免同bar污染
) -> Dict[str, Any]:

    # 1) 时间范围
    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023, 1, 1))

    # 2) 拉数据
    combo_data = signal_to_dataframe(
        data_dir, watermark_dir, venue, symbol,
        start_ms, end_ms, signal_module, base_dir
    )
    if combo_data is None or combo_data.empty:
        raise ValueError("combo_data is empty")

    combo_data = combo_data.sort_index()

    # 3) 价格序列：默认用 close；可选平滑（但会放大低频单调性）
    close = combo_data["close"].astype(float)
    if use_smoothed_price:
        price = close.rolling(window=rolling_window, min_periods=rolling_window).mean()
    else:
        price = close

    # 4) forward return（对齐到 t）
    ret_col = f"ret_{'rm'+str(rolling_window) if use_smoothed_price else 'close'}_fwd{horizon}"
    combo_data[ret_col] = _forward_return(price, horizon=horizon)

    # 5) 因子列（signal + factor*）
    factors = ["signal"] + [c for c in combo_data.columns if c.startswith("factor")]
    factors = [c for c in factors if c in combo_data.columns and combo_data[c].notna().any()]
    if not factors:
        raise ValueError("No valid factors found (signal/factor* all empty).")

    # 6) 对齐 + dropna
    aligned = combo_data[factors + [ret_col]].dropna()
    if aligned.empty:
        raise ValueError("No overlapping non-NaN samples for factors and forward return.")

    y = aligned[ret_col].astype(float)

    # 7) rolling IC
    rolling_ic_df = pd.DataFrame(index=aligned.index)
    used_factors = []

    for f in factors:
        x = aligned[f].astype(float)

        # ✅ 关键：滞后一根，避免同bar信息污染
        if lag_factors_by_1:
            x = x.shift(1)

        tmp = pd.concat([x, y], axis=1).dropna()
        if tmp.shape[0] < ic_window:
            continue

        x2 = tmp.iloc[:, 0]
        y2 = tmp.iloc[:, 1]

        if method.lower() == "pearson":
            ic = x2.rolling(window=ic_window, min_periods=ic_window).corr(y2)
        elif method.lower() == "spearman":
            # rolling spearman（rank corr）
            ic = _rolling_spearman_ic(x2, y2, window=ic_window, min_periods=ic_window)
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")

        rolling_ic_df[f] = ic
        used_factors.append(f)

    if rolling_ic_df.empty:
        raise ValueError("rolling_ic_df is empty (not enough data after alignment / windowing).")

    # 8) plotting（总图）
    out_paths: List[str] = []

    fig, ax = plt.subplots(figsize=(12, 6))
    rolling_ic_df.plot(ax=ax)
    ax.axhline(0.0, linewidth=1)
    ax.set_title(
        f"Rolling IC ({method.title()}) - {venue or ''} {symbol or ''} | "
        f"{start.date() if start else '2022-01-01'} to {end.date() if end else '2023-01-01'} | "
        f"ret_h={horizon}, ic_w={ic_window}, "
        f"{'smoothed_price' if use_smoothed_price else 'close_price'}, lag1={lag_factors_by_1}"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling IC")
    ax.legend(title="Factor", loc="best")
    fig.tight_layout()

    if base_dir:
        out_dir = Path(base_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        all_path = out_dir / "rolling_ic_all_factors.png"
        fig.savefig(all_path, dpi=150)
        out_paths.append(all_path.name)

    plt.close(fig)

    # 9) plotting（单因子图）
    if base_dir:
        out_dir = Path(base_dir)
        for f in used_factors:
            fig, ax = plt.subplots(figsize=(12, 6))
            rolling_ic_df[f].plot(ax=ax)
            ax.axhline(0.0, linewidth=1)
            ax.set_title(
                f"Rolling IC ({method.title()}) - {venue or ''} {symbol or ''} | factor={f} | "
                f"{start.date() if start else '2022-01-01'} to {end.date() if end else '2023-01-01'} | "
                f"ret_h={horizon}, ic_w={ic_window}, "
                f"{'smoothed_price' if use_smoothed_price else 'close_price'}, lag1={lag_factors_by_1}"
            )
            ax.set_xlabel("Time")
            ax.set_ylabel("Rolling IC")
            fig.tight_layout()

            fpath = out_dir / f"rolling_ic_{f}.png"
            fig.savefig(fpath, dpi=150)
            out_paths.append(fpath.name)
            plt.close(fig)

    return {
        "rolling_ic_curve": out_paths if base_dir else [],
        "rolling_ic_table": rolling_ic_df,
        "meta": {
            "method": method,
            "horizon": horizon,
            "ic_window": ic_window,
            "use_smoothed_price": use_smoothed_price,
            "rolling_window": rolling_window if use_smoothed_price else None,
            "lag_factors_by_1": lag_factors_by_1,
            "factors": used_factors,
            "ret_col": ret_col,
        }
    }
