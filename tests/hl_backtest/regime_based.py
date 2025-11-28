import numpy as np
import pandas as pd

# 使用 data_store 导出的 1m K 线，默认列名：
# - "t": 时间戳
# - "c": 收盘价
# - "v": 成交量
PRICE_COL = "c"
VOL_COL = "v"
MAX_LEVERAGE = 5.0

LAST_BAND = 0   # 可以给其它策略用
_SIGNAL_HISTORY = []

# ==== 最小持仓计数器（每个 baseline 一套） ====
MA_LAST_SIDE = 0
MA_HOLD_BARS = 0

Z_LAST_SIDE = 0
Z_HOLD_BARS = 0

TS_LAST_SIDE = 0
TS_HOLD_BARS = 0


def reset_signal_history() -> None:
    """清空信号历史（在每次回测前调用）。"""
    _SIGNAL_HISTORY.clear()


def get_signal_df() -> pd.DataFrame:
    """将当前缓存的信号历史导出为 DataFrame。"""
    if not _SIGNAL_HISTORY:
        return pd.DataFrame()
    df = pd.DataFrame(_SIGNAL_HISTORY)
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    return df


def band_from_signal(x: float) -> int:
    if x >= 0.4:
        return 1
    if x <= -0.4:
        return -1
    return 0


# ============================================================
#             几个常见的基线（Baseline）策略
# ============================================================

def bh_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,   # 为了接口统一，这里加上但实际上 BH 始终满仓
) -> float:
    """
    基线 1：Buy & Hold（永远做多）

    - 有数据且权益>0 就一直返回 +1.0（满仓多）
    - 用作最简单的方向性 Beta benchmark
    """
    if df is None or df.empty or current_equity <= 0:
        return 0.0
    return 1.0


def ma_crossover_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,
    FAST_WINDOW: int = 20,
    SLOW_WINDOW: int = 60,
) -> float:
    """
    基线 2：MA 交叉趋势策略（1m）

    - 快线 > 慢线 → +1.0（多）
    - 快线 < 慢线 → -1.0（空）
    - 其余 → 0.0（空仓）

    MIN_HOLD_BARS:
      - 当前有仓位时，如果尚未持有足够 bar 数，则不允许平仓 / 反向，
        直接维持 current_pos。
    """
    global MA_LAST_SIDE, MA_HOLD_BARS

    if df is None or df.empty or current_equity <= 0 or PRICE_COL not in df.columns:
        return 0.0

    if len(df) < max(FAST_WINDOW, SLOW_WINDOW):
        return 0.0

    closes = df[PRICE_COL].astype(float)
    fast_ma = closes.rolling(FAST_WINDOW).mean().iloc[-1]
    slow_ma = closes.rolling(SLOW_WINDOW).mean().iloc[-1]

    if np.isnan(fast_ma) or np.isnan(slow_ma):
        return 0.0

    # 原始目标方向（不考虑最小持仓）
    if fast_ma > slow_ma * 1.0001:
        raw_side = 1.0
    elif fast_ma < slow_ma * 0.9999:
        raw_side = -1.0
    else:
        raw_side = 0.0

    # === 最小持仓周期逻辑 ===
    curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

    if curr_side == 0:
        # 空仓 → 重置计数
        MA_HOLD_BARS = 0
    else:
        # 有仓位
        if curr_side == MA_LAST_SIDE:
            MA_HOLD_BARS += 1
        else:
            MA_HOLD_BARS = 1
            MA_LAST_SIDE = curr_side

    if MIN_HOLD_BARS > 0 and curr_side != 0:
        # 有仓 & 未达到最小持仓 → 即便信号想平仓/反向，也先 hold
        if raw_side != curr_side and MA_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)

    return raw_side


def zscore_mr_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,
    WINDOW: int = 40,
    ENTRY_Z: float = 1.5,
    EXIT_Z: float = 0.3,
) -> float:
    """
    基线 3：Z-score 均值回归（1m）

    - z = (P - MA) / std
    - z >  ENTRY_Z  → -1.0（做空）
    - z < -ENTRY_Z  → +1.0（做多）
    - |z| < EXIT_Z  → 0.0（平仓）

    MIN_HOLD_BARS:
      - 持仓期间未达到 bar 数，信号即使满足平仓或反手也先 hold。
    """
    global Z_LAST_SIDE, Z_HOLD_BARS

    if df is None or df.empty or current_equity <= 0 or PRICE_COL not in df.columns:
        return 0.0

    if len(df) < WINDOW:
        return 0.0

    closes = df[PRICE_COL].astype(float)
    ma = closes.rolling(WINDOW).mean()
    std = closes.rolling(WINDOW).std(ddof=0)

    mu = ma.iloc[-1]
    sigma = std.iloc[-1]
    p = closes.iloc[-1]

    if np.isnan(mu) or np.isnan(sigma) or sigma < 1e-12:
        return 0.0

    z = (p - mu) / sigma

    # 原始方向
    if abs(z) < EXIT_Z:
        raw_side = 0.0
    elif z > ENTRY_Z:
        raw_side = -1.0
    elif z < -ENTRY_Z:
        raw_side = 1.0
    else:
        # 中间区域：保持现有仓位
        raw_side = float(current_pos)

    # === 最小持仓周期逻辑 ===
    curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

    if curr_side == 0:
        Z_HOLD_BARS = 0
    else:
        if curr_side == Z_LAST_SIDE:
            Z_HOLD_BARS += 1
        else:
            Z_HOLD_BARS = 1
            Z_LAST_SIDE = curr_side

    if MIN_HOLD_BARS > 0 and curr_side != 0:
        # 有仓 & 未达到最小持仓 → 不允许从有仓 → 空仓 或翻向
        if (raw_side == 0.0 or np.sign(raw_side) != curr_side) and Z_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)

    return raw_side


def tsmom_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,
    LOOKBACK: int = 50,
    THRESHOLD: float = 0.0,
) -> float:
    """
    基线 4：时间序列动量（TSMOM，1m）

    - ret = P_t / P_{t-k} - 1
    - ret >  THRESHOLD → +1.0（多）
    - ret < -THRESHOLD → -1.0（空）
    - |ret| <= THRESHOLD → 0.0（空仓）

    MIN_HOLD_BARS:
      - 同样在有仓时要求至少持有这么多个 bar 才能平仓/反向。
    """
    global TS_LAST_SIDE, TS_HOLD_BARS

    if df is None or df.empty or current_equity <= 0 or PRICE_COL not in df.columns:
        return 0.0

    if len(df) <= LOOKBACK:
        return 0.0

    closes = df[PRICE_COL].astype(float)
    p_now = closes.iloc[-1]
    p_past = closes.iloc[-(LOOKBACK + 1)]

    if np.isnan(p_now) or np.isnan(p_past) or p_past <= 0:
        return 0.0

    ret = p_now / p_past - 1.0

    # 原始方向
    if ret > THRESHOLD:
        raw_side = 1.0
    elif ret < -THRESHOLD:
        raw_side = -1.0
    else:
        raw_side = 0.0

    # === 最小持仓逻辑 ===
    curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

    if curr_side == 0:
        TS_HOLD_BARS = 0
    else:
        if curr_side == TS_LAST_SIDE:
            TS_HOLD_BARS += 1
        else:
            TS_HOLD_BARS = 1
            TS_LAST_SIDE = curr_side

    if MIN_HOLD_BARS > 0 and curr_side != 0:
        if raw_side != curr_side and TS_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)

    return raw_side
