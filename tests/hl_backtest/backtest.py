import math
from typing import Callable, Dict, Any

import numpy as np
import pandas as pd

def interval_to_minutes(interval: str) -> float:
    """
    将 K 线周期字符串转换为“每根 K 线多少分钟”。

    支持：
      - "1m","5m","15m","30m"
      - "1h","2h","4h"
      - "1d"（按 24h 算）
    """
    interval = interval.strip().lower()
    if interval.endswith("m"):
        return float(int(interval[:-1]))  # 例如 "5m" -> 5
    if interval.endswith("h"):
        return float(int(interval[:-1]) * 60)  # "1h" -> 60
    if interval.endswith("d"):
        return float(int(interval[:-1]) * 24 * 60)  # "1d" -> 1440
    raise ValueError(f"Unsupported interval: {interval}")

def run_backtest(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, float, float], float],
    initial_equity: float = 1000.0,
    max_leverage: float = 5.0,
    taker_fee_rate: float = 0.0005,
    slippage_bps: float = 1.0,
    price_col: str = "c",
    start_index: int = 50,
    interval: str = "1m",
) -> Dict[str, Any]:
    """
    非 OOP 的简单永续合约回测引擎：
    - 每根 K 线收盘时刻调用 strategy_fn
    - 线性合约 PnL: (P_t - P_{t-1}) * position
    - 按 max_leverage 做简单杠杆约束
    """

    df = df.reset_index(drop=True).copy()
    closes = df[price_col].values.astype(float)
    times = df["T"].values  # 收盘时间

    n = len(df)
    if n < start_index + 2:
        raise ValueError("数据太短，start_index 设得过大。")

    equity = float(initial_equity)
    position = 0.0
    last_price = float(closes[0])

    slippage = slippage_bps / 10000.0

    equity_list = [equity] * n
    pos_list = [0.0] * n

    trades = []  # 用列表装 dict，最后变 DataFrame

    for i in range(1, n):
        price = float(closes[i])
        t = pd.to_datetime(times[i])

        # 1) 先结算上一根 ~ 当前价格区间的 PnL
        pnl = (price - last_price) * position
        equity += pnl

        # 2) 让策略决策
        if i >= start_index:
            df_slice = df.iloc[: i + 1].copy()
            target_pos = float(strategy_fn(df_slice, position, equity))

            # 杠杆约束：|pos * price| <= max_leverage * equity
            if price > 0 and equity > 0 and max_leverage > 0:
                max_pos = max_leverage * equity / price
                target_pos = max(-max_pos, min(max_pos, target_pos))

            # 3) 调仓：收手续费 + 滑点
            delta_pos = target_pos - position
            if abs(delta_pos) > 1e-9:
                if delta_pos > 0:
                    fill_price = price * (1 + slippage)
                else:
                    fill_price = price * (1 - slippage)

                notional = abs(delta_pos) * fill_price
                fee = notional * taker_fee_rate

                equity -= fee
                trades.append(
                    {
                        "index": i,
                        "time": t,
                        "price": fill_price,
                        "old_pos": position,
                        "new_pos": target_pos,
                        "fee": fee,
                        "equity_after": equity,
                    }
                )
                position = target_pos

        equity_list[i] = equity
        pos_list[i] = position
        last_price = price

    equity_series = pd.Series(equity_list, index=df["T"], name="equity")
    pos_series = pd.Series(pos_list, index=df["T"], name="position")
    trades_df = pd.DataFrame(trades)

    stats = _calc_stats(equity_series, interval=interval)

    return {
        "equity_curve": equity_series,
        "position_series": pos_series,
        "trades": trades_df,
        "stats": stats,
    }



def _calc_stats(equity_curve: pd.Series, interval: str = "1m") -> Dict[str, float]:
    ret = equity_curve.pct_change().fillna(0.0)

    first = float(equity_curve.iloc[0])
    last = float(equity_curve.iloc[-1])
    if first == 0.0:
        total_return = float("nan")
    else:
        total_return = last / first - 1.0

    # 根据 interval 自动计算年化因子
    # 例如:
    #   "1m" -> bar_minutes = 1   -> 每天 1440 根
    #   "5m" -> bar_minutes = 5   -> 每天 288 根
    #   "1h" -> bar_minutes = 60  -> 每天 24 根
    bar_minutes = interval_to_minutes(interval)
    freq_per_day = (24 * 60) / bar_minutes
    annual_factor = math.sqrt(365 * freq_per_day)

    vol = ret.std(ddof=0)
    sharpe = (ret.mean() * annual_factor / vol) if vol > 0 else float("nan")

    cummax = equity_curve.cummax()
    drawdown = equity_curve / cummax - 1.0
    max_dd = float(drawdown.min())

    return {
        "final_equity": last,
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
    }
