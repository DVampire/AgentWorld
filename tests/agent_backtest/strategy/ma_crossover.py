"""MA Crossover baseline strategy."""

import numpy as np
import pandas as pd
from .base import PRICE_COL

# ==== 最小持仓计数器 ====
MA_LAST_SIDE = 0
MA_HOLD_BARS = 0


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

