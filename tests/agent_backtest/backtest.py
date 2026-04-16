import math
from typing import Callable, Dict, Any, Optional

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame, float, float], float],
    initial_equity: float = 1000.0,
    max_leverage: float = 5.0,
    taker_fee_rate: float = 0.00045,
    slippage_bps: float = 1.0,
    price_col: str = "c",
    start_index: int = 50,
) -> Dict[str, Any]:
    """
    简化版回测引擎（假定使用 1m K 线）：

    - 策略函数返回的是仓位比例 proportion ∈ [-1, 1]
        proportion =  1.0 → 最大多头
        proportion = -1.0 → 最大空头
        proportion =  0.0 → 空仓

    - 实际张数 = proportion * equity * max_leverage / price
    - 每根 K 线先按持仓做一次 mark-to-market，再根据策略信号调仓
    - 手续费:
        fee = |Δpos| * fill_price * taker_fee_rate

    现在每笔交易记录中增加：
        segment_pnl     : 上一次调仓后到本次调仓前，这一段持仓产生的总盈亏（不含本次手续费）
        realized_pnl    : 已实现盈亏（= segment_pnl - fee）
        realized_pnl_cum: 截至当前这笔交易的累积已实现盈亏（净值）
    """
    df = df.reset_index(drop=True).copy()
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' 不在 df 列中")

    closes = df[price_col].astype(float).values
    times = pd.to_datetime(df["t"]).values if "t" in df.columns else df.index.to_numpy()

    equity: float = float(initial_equity)
    pos_frac: float = 0.0       # 仓位比例 ∈ [-1,1]
    position: float = 0.0       # 实际张数
    last_price: float = float(closes[0])

    slippage = slippage_bps / 10000.0

    equity_list = []
    pos_frac_list = []
    trades = []

    last_trade_equity: float = equity        # 上一次调仓之后的权益
    realized_pnl_cum: float = 0.0           # 累积已实现盈亏（净额）

    for i in range(len(df)):
        price = float(closes[i])
        time = times[i]

        # ---------- 1. 先按当前持仓记账（浮盈/浮亏） ----------
        # 永续合约盈亏计算：
        # - 做多（position > 0）：价格上涨时赚钱，价格下跌时亏钱
        # - 做空（position < 0）：价格下跌时赚钱，价格上涨时亏钱
        # 公式：pnl = (price - last_price) * position
        #   做多：position > 0, price上涨 → pnl > 0 ✓
        #   做空：position < 0, price下跌 → pnl > 0 ✓
        pnl = (price - last_price) * position
        equity += pnl
        last_price = price

        # ---------- 2. 调用策略，给出新的仓位比例 ----------
        if i >= start_index and price > 0 and equity > 0:
            new_frac = float(strategy_fn(df.iloc[: i + 1], pos_frac, equity))
            new_frac = float(max(-1.0, min(1.0, new_frac)))  # 限制在 [-1,1]
        else:
            new_frac = pos_frac

        # ---------- 3. 如果仓位比例变化，则调仓 ----------
        if abs(new_frac - pos_frac) > 1e-6 and price > 0 and equity > 0:
            equity_before_trade = equity

            # 计算目标持仓量（永续合约）：
            # - new_frac = 1.0 → target_position > 0（做多）
            # - new_frac = -1.0 → target_position < 0（做空）
            # - new_frac = 0.0 → target_position = 0（空仓）
            target_position = new_frac * equity * max_leverage / price
            delta = target_position - position

            if abs(delta) > 1e-9:
                side = "BUY" if delta > 0 else "SELL"
                # 加/减滑点
                fill = price * (1 + slippage if delta > 0 else 1 - slippage)

                notional = abs(delta) * fill
                fee = notional * taker_fee_rate

                # 本段持仓（上一次调仓之后到本次调仓之前）的 PnL（不含本次 fee）
                segment_pnl = equity_before_trade - last_trade_equity

                # 已实现盈亏 = 本段 PnL - 当前交易手续费
                realized_pnl = segment_pnl - fee
                realized_pnl_cum += realized_pnl

                equity_after_trade = equity_before_trade - fee

                # 交易动作分类
                if abs(pos_frac) < 1e-9 and abs(new_frac) > 1e-9:
                    action = "OPEN_LONG" if new_frac > 0 else "OPEN_SHORT"
                elif abs(new_frac) < 1e-9 and abs(pos_frac) > 1e-9:
                    action = "CLOSE"
                elif pos_frac * new_frac < 0:
                    action = "REVERSE"
                else:
                    action = "ADJUST"

                trades.append(
                    {
                        "index": i,
                        "time": time,
                        "price": fill,
                        "old_frac": pos_frac,
                        "new_frac": new_frac,
                        "old_pos": position,
                        "new_pos": target_position,
                        "delta_pos": delta,
                        "side": side,
                        "action": action,
                        "fee": fee,
                        "segment_pnl": segment_pnl,
                        "realized_pnl": realized_pnl,
                        "realized_pnl_cum": realized_pnl_cum,
                        "equity_before": equity_before_trade,
                        "equity_after": equity_after_trade,
                    }
                )

                equity = equity_after_trade
                last_trade_equity = equity_after_trade
                position = target_position
                pos_frac = new_frac

        equity_list.append(equity)
        pos_frac_list.append(pos_frac)

    equity_series = pd.Series(
        equity_list,
        index=df["t"] if "t" in df.columns else df.index,
        name="equity",
    )
    pos_frac_series = pd.Series(pos_frac_list, index=equity_series.index, name="pos_frac")
    trades_df = pd.DataFrame(trades)

    stats = _calc_stats(equity_series, trades_df)

    return {
        "equity_curve": equity_series,
        "pos_frac_series": pos_frac_series,
        "trades": trades_df,
        "stats": stats,
    }


def _calc_stats(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """
    简化版统计：
      - final_equity
      - total_return
      - sharpe（假设 1m K 线）
      - max_drawdown

    如果有 trades，额外统计：
      - num_trades
      - total_fees
      - avg_fee_per_trade
      - num_long_trades
      - num_short_trades
      - total_realized_pnl
      - unrealized_pnl
      - win_rate（以 realized_pnl > 0 判定）
    """
    stats: Dict[str, float] = {}

    if equity_curve is None or len(equity_curve) == 0:
        return stats

    first = float(equity_curve.iloc[0])
    last = float(equity_curve.iloc[-1])
    ret = equity_curve.pct_change().fillna(0.0)

    total_return = (last / first - 1.0) if first > 0 else float("nan")

    # 1m K 线 → 一天 1440 根
    bars_per_day = 1440.0
    annual_factor = math.sqrt(365.0 * bars_per_day)

    vol = float(ret.std(ddof=0))
    sharpe = (ret.mean() * annual_factor / vol) if vol > 0 else float("nan")

    cummax = equity_curve.cummax()
    drawdown = (equity_curve / cummax - 1.0).fillna(0.0)
    max_dd = float(drawdown.min())

    stats.update(
        dict(
            final_equity=last,
            total_return=total_return,
            sharpe=sharpe,
            max_drawdown=max_dd,
        )
    )

    if trades is not None and not trades.empty:
        num_trades = int(len(trades))
        total_fees = float(trades["fee"].sum()) if "fee" in trades.columns else 0.0
        avg_fee_per_trade = float(total_fees / num_trades) if num_trades > 0 else 0.0

        num_long_trades = 0
        num_short_trades = 0
        if "action" in trades.columns:
            num_long_trades = int((trades["action"] == "OPEN_LONG").sum())
            num_short_trades = int((trades["action"] == "OPEN_SHORT").sum())

        # 总已实现盈亏
        total_realized_pnl = 0.0
        if "realized_pnl" in trades.columns:
            total_realized_pnl = float(trades["realized_pnl"].sum())

        # 总权益变动 = final - initial
        total_equity_pnl = last - first
        unrealized_pnl = total_equity_pnl - total_realized_pnl

        # 胜率：以 realized_pnl > 0 判定
        win_rate = float("nan")
        if "realized_pnl" in trades.columns and "action" in trades.columns:
            closed = trades[trades["action"].isin(["CLOSE", "REVERSE", "ADJUST"])]
            if not closed.empty:
                wins = (closed["realized_pnl"] > 0).sum()
                win_rate = float(wins / len(closed))

        stats.update(
            dict(
                num_trades=num_trades,
                total_fees=total_fees,
                avg_fee_per_trade=avg_fee_per_trade,
                num_long_trades=num_long_trades,
                num_short_trades=num_short_trades,
                total_realized_pnl=total_realized_pnl,
                unrealized_pnl=unrealized_pnl,
                win_rate=win_rate,
            )
        )

    return stats
