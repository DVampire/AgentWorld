"""Buy & Hold baseline strategy."""

import pandas as pd
from .base import PRICE_COL


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

