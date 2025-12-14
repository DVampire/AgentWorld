"""Z-score Mean Reversion baseline strategy."""

import numpy as np
import pandas as pd
from .base import PRICE_COL

# ==== 最小持仓计数器 ====
Z_LAST_SIDE = 0
Z_HOLD_BARS = 0


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

