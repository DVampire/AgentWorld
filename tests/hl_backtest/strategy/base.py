"""Base constants and utility functions for strategies."""

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
    """将信号值转换为波段值。"""
    if x >= 0.4:
        return 1
    if x <= -0.4:
        return -1
    return 0

