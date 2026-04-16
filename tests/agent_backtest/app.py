import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from typing import Dict, Callable
import os

from backtest import run_backtest
from regime_based import (
    PRICE_COL,
    MAX_LEVERAGE,
    bh_baseline,
    ma_crossover_baseline,
    zscore_mr_baseline,
    tsmom_baseline,
    livetrading_baseline,
    funding_arb_baseline,
    carry_momentum_baseline,
)

# ========= Streamlit 基本设置 =========
st.set_page_config(
    page_title="Crypto Benchmark Backtest",
    layout="wide",
    page_icon="📈",
)

st.title("📈 Crypto Benchmark Backtest Dashboard")

st.markdown(
    """
本面板会：
- 载入 1m K 线数据
- 同时回测 3 个策略（BH / LiveTrading）
- 支持为 **每个策略设置最小持仓周期（bar 数）**
- 展示 equity 对比 + stats 表格
- 一键导出 PDF 报告
"""
)

# ========= 侧边栏参数 =========
st.sidebar.header("基础参数")

coin = st.sidebar.text_input("合约标的 (COIN)", "BTC")
interval = st.sidebar.selectbox("K线周期", ["1m"], index=0)
lookback = st.sidebar.slider("Lookback candles", 1000, 20000, 5000, step=1000)

initial_equity = st.sidebar.number_input("初始资金", value=500.0, min_value=10.0, step=100.0)
taker_fee = st.sidebar.number_input("Taker 手续费", value=0.00045, format="%.5f")
slippage_bps = st.sidebar.number_input("滑点 (bps)", value=1.0, step=0.5)

use_testnet = st.sidebar.checkbox("使用 Testnet 数据", value=False)
local_only = st.sidebar.checkbox("仅使用本地缓存数据 (local)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("最小持仓周期 (bars)")

bh_min_hold = st.sidebar.number_input("BH min hold", min_value=0, value=0, step=1)
ma_min_hold = st.sidebar.number_input("MA min hold", min_value=0, value=0, step=1)
zs_min_hold = st.sidebar.number_input("ZScoreMR min hold", min_value=0, value=0, step=1)
ts_min_hold = st.sidebar.number_input("TSMOM min hold", min_value=0, value=0, step=1)
lt_gpt5_min_hold = st.sidebar.number_input("LiveTrading min hold", min_value=0, value=0, step=1)
fa_min_hold = st.sidebar.number_input("FundingArb min hold", min_value=0, value=0, step=1)
cm_min_hold = st.sidebar.number_input("CarryMomentum min hold", min_value=0, value=0, step=1)

run_button = st.sidebar.button("🚀 运行回测")


# ========= 工具函数 =========
@st.cache_data(show_spinner="加载 K 线数据中...")
def load_data(coin: str, interval: str, lookback: int, use_testnet: bool, local: bool) -> pd.DataFrame:
    """
    Load crypto price data from jsonl files in workdir.
    
    Args:
        coin: Coin symbol (e.g., "BTC")
        interval: K-line interval (e.g., "1day", "1m")
        lookback: Number of candles to load
        use_testnet: Not used (kept for compatibility)
        local: Not used (kept for compatibility)
    
    Returns:
        DataFrame with columns: t, o, h, l, c, v
    """
    # Map interval to directory name
    interval_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "1d": "1day",
        "1day": "1day",
    }
    
    # Get project root (assuming tests/hl_backtest/app.py is in tests/hl_backtest/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Build file path
    interval_dir = interval_map.get(interval, interval)
    symbol = f"{coin}USDT"
    file_path = os.path.join(
        project_root,
        "workdir",
        "crypto",
        f"crypto_binance_price_{interval_dir}",
        f"{symbol}.jsonl"
    )
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load jsonl file
    df = pd.read_json(file_path, lines=True)
    
    # Rename columns to match backtest format
    # timestamp -> t, open -> o, high -> h, low -> l, close -> c, volume -> v
    column_map = {
        "timestamp": "t",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
    }
    
    df = df.rename(columns=column_map)
    
    # Ensure timestamp is datetime
    df["t"] = pd.to_datetime(df["t"])
    
    # Sort by timestamp
    df = df.sort_values("t").reset_index(drop=True)
    
    # Apply lookback limit (take last N rows)
    if lookback > 0 and len(df) > lookback:
        df = df.tail(lookback).reset_index(drop=True)
    
    # Ensure numeric columns are float
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = df[col].astype(float)
    
    return df


def run_all_strategies(
    df: pd.DataFrame,
    strategies: Dict[str, Callable[[pd.DataFrame, float, float], float]],
    initial_equity: float,
    taker_fee: float,
    slippage_bps: float,
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for name, fn in strategies.items():
        result = run_backtest(
            df=df,
            strategy_fn=fn,
            initial_equity=initial_equity,
            max_leverage=MAX_LEVERAGE,
            taker_fee_rate=taker_fee,
            slippage_bps=slippage_bps,
            price_col=PRICE_COL,
            start_index=100,
        )
        results[name] = result
    return results


def create_pdf_report(
    coin: str,
    interval: str,
    df: pd.DataFrame,
    results: Dict[str, Dict],
    stats_df: pd.DataFrame,
) -> bytes:
    """
    使用 Matplotlib PdfPages 生成 PDF：
      - Page1: equity 对比图
      - Page2: stats 表格
    """
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: Equity comparison
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        for name, res in results.items():
            eq = res["equity_curve"]
            ax1.plot(eq.index, eq.values, label=name)
        ax1.set_title(f"Equity Comparison - {coin} {interval}")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Equity")
        ax1.grid(True)
        ax1.legend()
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2: Stats table
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.axis("off")
        tbl = ax2.table(
            cellText=stats_df.round(4).values,
            colLabels=stats_df.columns,
            rowLabels=stats_df.index,
            loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1.0, 1.2)
        ax2.set_title("Stats Summary", fontsize=14, pad=20)
        pdf.savefig(fig2)
        plt.close(fig2)

    buffer.seek(0)
    return buffer.getvalue()


# ========= 主逻辑 =========
if not run_button:
    st.info("在左侧设置参数后，点击 **🚀 运行回测**。")
else:
    # 1) 载入数据
    df = load_data(coin, interval, lookback, use_testnet, local_only)
    st.success(f"数据加载完成：{coin} {interval}, 共 {len(df)} 根K线。")
    st.write(
        f"时间范围：**{df['t'].iloc[0]}** → **{df['t'].iloc[-1]}**"
    )

    # 2) 构造带 min-hold 的策略封装
    def make_bh_strat(min_hold: int):
        def strat(df_, pos_, eq_):
            return bh_baseline(df_, pos_, eq_, MIN_HOLD_BARS=int(min_hold))
        return strat

    def make_ma_strat(min_hold: int):
        def strat(df_, pos_, eq_):
            return ma_crossover_baseline(df_, pos_, eq_, MIN_HOLD_BARS=int(min_hold))
        return strat

    def make_zs_strat(min_hold: int):
        def strat(df_, pos_, eq_):
            return zscore_mr_baseline(df_, pos_, eq_, MIN_HOLD_BARS=int(min_hold))
        return strat

    def make_ts_strat(min_hold: int):
        def strat(df_, pos_, eq_):
            return tsmom_baseline(df_, pos_, eq_, MIN_HOLD_BARS=int(min_hold))
        return strat

    def make_lt_strat(min_hold: int):
        def strat(df_, pos_, eq_):
            return livetrading_baseline(df_, pos_, eq_, MIN_HOLD_BARS=int(min_hold))
        return strat

    def make_fa_strat(min_hold: int):
        def strat(df_, pos_, eq_):
            return funding_arb_baseline(df_, pos_, eq_, MIN_HOLD_BARS=int(min_hold))
        return strat

    def make_cm_strat(min_hold: int):
        def strat(df_, pos_, eq_):
            return carry_momentum_baseline(df_, pos_, eq_, MIN_HOLD_BARS=int(min_hold))
        return strat

    strategies: Dict[str, Callable[[pd.DataFrame, float, float], float]] = {
        "BH": make_bh_strat(bh_min_hold),
        "MA": make_ma_strat(ma_min_hold),
        "ZScoreMR": make_zs_strat(zs_min_hold),
        "TSMOM": make_ts_strat(ts_min_hold),
        "livetrading": make_lt_strat(lt_gpt5_min_hold),
        "FundingArb": make_fa_strat(fa_min_hold),
        "CarryMomentum": make_cm_strat(cm_min_hold),
    }

    # 3) 跑所有策略
    results = run_all_strategies(df, strategies, initial_equity, taker_fee, slippage_bps)

    # 4) Equity 对比图
    st.subheader("Equity 曲线对比")

    fig, ax = plt.subplots(figsize=(11, 5))
    for name, res in results.items():
        eq = res["equity_curve"]
        ax.plot(eq.index, eq.values, label=name)
    ax.set_title(f"Equity Comparison ({coin} {interval})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # 5) Stats 表格
    st.subheader("Stats Summary")

    stats_rows = []
    for name, res in results.items():
        s = res["stats"].copy()
        s["name"] = name
        stats_rows.append(s)
    stats_df = pd.DataFrame(stats_rows).set_index("name")

    st.dataframe(stats_df.style.format("{:.4f}"))

    # 6) 导出 PDF
    st.subheader("导出报告")

    pdf_bytes = create_pdf_report(coin, interval, df, results, stats_df)

    st.download_button(
        label="📄 下载 PDF 报告",
        data=pdf_bytes,
        file_name=f"{coin}_{interval}_benchmark_report.pdf",
        mime="application/pdf",
    )

    st.caption("PDF 中包含：Equity 对比图 + Stats Summary 表格。")
