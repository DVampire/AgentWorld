"""Time Series Momentum baseline strategy."""

import numpy as np
import pandas as pd
from .base import PRICE_COL

# ==== 最小持仓计数器 ====
TS_LAST_SIDE = 0
TS_HOLD_BARS = 0


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

