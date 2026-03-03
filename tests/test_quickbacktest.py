from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)
from src.environment.quickbacktest.backtest import backtest_strategy,STRATEGY_PARAMS,COMMISSION
from src.environment.quickbacktest.base_types import BaseStrategy,BaseSignal
import pandas as pd
from typing import Literal, Tuple, Union, List, Dict, Any
import numpy as np
import backtrader as bt
from src.environment.quickbacktest.utils import (
    get_strategy_sharpe_ratio,
    get_strategy_cumulative_return,
    get_strategy_maxdrawdown,
    get_strategy_total_commission,
    get_strategy_win_rate,
    plot_cumulative_return
)



async def backtest(
        self,
        data: pd.DataFrame,
        code: Union[str, List[str]],
        strategy: BaseStrategy,
        signal: BaseSignal,
        strategy_kwargs: Dict = STRATEGY_PARAMS,
        commission_kwargs: Dict = COMMISSION,
    ) -> Any:
        """Run backtest"""
        combo_data = signal.fit(data)
        result = backtest_strategy(
            data=combo_data,
            code=code,
            strategy=strategy,
            strategy_kwargs=strategy_kwargs,
            commission_kwargs=commission_kwargs,
        )

        return {
            "sharpe_ratio": get_strategy_sharpe_ratio(result.cerebro),
            "cumulative_return": get_strategy_cumulative_return(result.cerebro).iloc[-1],
            "max_drawdown": get_strategy_maxdrawdown(result.cerebro),
        }


class FundingNoiseArea(BaseSignal):
    """
    Funding-NoiseArea（Perps）

    输出列（长表）：
        - signal_1: UpperBound
        - signal_2: LowerBound
        - signal_3: close
        - vwap: rolling VWAP（BaseSignal.calculate_rolling_vwap，单独一列）

    BaseSignal.REQUIRED = ("get_signals", "concat_signals")
    """

    def __init__(self, ohlcv: pd.DataFrame) -> None:
        super().__init__(ohlcv)

    # ============================================================
    # Funding-cycle utilities
    # ============================================================
    def _cycle_id_pos(
        self,
        funding_hours: int = 8,
        tz: str = "Asia/Shanghai",
        assume_naive_is_utc: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算对齐到 tz 的 funding 周期 cycle_id 与周期内位置 pos_in_cycle（以 bar 序号计）

        返回：
            cycle_id: np.ndarray(len(index))
            pos_in_cycle: np.ndarray(len(index))
        """
        idx = self.close.index
        ts = idx

        # 统一到 tz-aware
        if ts.tz is None:
            ts = ts.tz_localize("UTC" if assume_naive_is_utc else tz)
        ts_local = ts.tz_convert(tz)

        # 本地日零点
        day0 = ts_local.normalize()

        # 距离本地日零点的秒数
        sec_from_day0 = (ts_local.view("int64") - day0.view("int64")) // 10**9

        funding_seconds = int(funding_hours * 3600)
        cycle_in_day = (sec_from_day0 // funding_seconds).astype(np.int64)

        # 跨天索引（递增整数）
        day_index = day0.date.astype("datetime64[D]").astype(np.int64)

        cycles_per_day = int(24 // funding_hours)
        cycle_id = day_index * cycles_per_day + cycle_in_day

        # 周期内位置：同 cycle_id 内 cumcount
        pos_in_cycle = (
            pd.Series(cycle_id, index=idx)
            .groupby(cycle_id)
            .cumcount()
            .values.astype(np.int64)
        )

        return cycle_id, pos_in_cycle

    def calculate_funding_anchor(
        self,
        funding_hours: int = 8,
        tz: str = "Asia/Shanghai",
        assume_naive_is_utc: bool = True,
    ) -> pd.DataFrame:
        """
        A_k：每个 funding 周期第一根 bar 的 close，并在周期内 forward-fill
        """
        cycle_id, _ = self._cycle_id_pos(
            funding_hours=funding_hours, tz=tz, assume_naive_is_utc=assume_naive_is_utc
        )

        anchor = self.close.copy()
        for c in anchor.columns:
            anchor[c] = anchor[c].groupby(cycle_id).transform("first")
        return anchor

    def calculate_sigma(
        self,
        window: int = 14,
        mode: Literal["mean", "quantile"] = "mean",
        q: float = 0.85,
        funding_hours: int = 8,
        tz: str = "Asia/Shanghai",
        use_shift: bool = True,
        assume_naive_is_utc: bool = True,
    ) -> pd.DataFrame:
        """
        sigma：对相同 pos_in_cycle 的 move=|close/anchor-1| 做滚动统计（跨周期）
        use_shift=True：对每个 pos 序列 shift(1) 避免同周期污染
        """
        idx = self.close.index

        anchor = self.calculate_funding_anchor(
            funding_hours=funding_hours, tz=tz, assume_naive_is_utc=assume_naive_is_utc
        )
        move = (self.close.div(anchor) - 1.0).abs()

        _, pos_in_cycle = self._cycle_id_pos(
            funding_hours=funding_hours, tz=tz, assume_naive_is_utc=assume_naive_is_utc
        )
        pos = pd.Series(pos_in_cycle, index=idx)

        sigma = pd.DataFrame(index=idx, columns=self.close.columns, dtype=float)

        for c in self.close.columns:
            s = move[c].astype(float)

            if mode == "mean":
                out = s.groupby(pos, group_keys=False).apply(
                    lambda x: x.rolling(window=window, min_periods=window).mean()
                )
            elif mode == "quantile":
                out = s.groupby(pos, group_keys=False).apply(
                    lambda x: x.rolling(window=window, min_periods=window).quantile(q)
                )
            else:
                raise ValueError("mode must be 'mean' or 'quantile'")

            if use_shift:
                out = out.groupby(pos, group_keys=False).shift(1)

            sigma[c] = out.values

        return sigma

    def calculate_bound(
        self,
        window: int = 14,
        method: Literal["U", "L"] = "U",
        mode: Literal["mean", "quantile"] = "mean",
        q: float = 0.85,
        funding_hours: int = 8,
        tz: str = "Asia/Shanghai",
        use_shift: bool = True,
        assume_naive_is_utc: bool = True,
    ) -> pd.DataFrame:
        """
        Upper/Lower:
            U: anchor * (1 + sigma)
            L: anchor * (1 - sigma)
        """
        anchor = self.calculate_funding_anchor(
            funding_hours=funding_hours, tz=tz, assume_naive_is_utc=assume_naive_is_utc
        )
        sigma = self.calculate_sigma(
            window=window,
            mode=mode,
            q=q,
            funding_hours=funding_hours,
            tz=tz,
            use_shift=use_shift,
            assume_naive_is_utc=assume_naive_is_utc,
        )

        m = method.upper()
        if m == "U":
            return anchor.mul(1.0 + sigma)
        if m == "L":
            return anchor.mul(1.0 - sigma)
        raise ValueError("method must be 'U' or 'L'")

    # ============================================================
    # REQUIRED API
    # ============================================================
    def get_signals(
        self,
        window: int = 14,
        mode: Literal["mean", "quantile"] = "quantile",
        q: float = 0.85,
        funding_hours: int = 8,
        tz: str = "Asia/Shanghai",
        use_shift: bool = True,
        assume_naive_is_utc: bool = True,
        vwap_window: int | None = None,
    ) -> pd.DataFrame:
        """
        返回长表：
            trade_time, code, signal_1, signal_2, signal_3, vwap

        定义：
            signal_1 = UpperBound
            signal_2 = LowerBound
            signal_3 = close
            vwap     = BaseSignal.calculate_rolling_vwap(window=vwap_window or window)
        """
        upper = self.calculate_bound(
            window=window,
            method="U",
            mode=mode,
            q=q,
            funding_hours=funding_hours,
            tz=tz,
            use_shift=use_shift,
            assume_naive_is_utc=assume_naive_is_utc,
        )
        lower = self.calculate_bound(
            window=window,
            method="L",
            mode=mode,
            q=q,
            funding_hours=funding_hours,
            tz=tz,
            use_shift=use_shift,
            assume_naive_is_utc=assume_naive_is_utc,
        )
        price = self.close.copy()

        vw = self.calculate_rolling_vwap(window=(vwap_window or window))  # 宽表：index=time, columns=code

        out = pd.concat(
            [
                upper.stack().to_frame("signal_1"),
                lower.stack().to_frame("signal_2"),
                price.stack().to_frame("signal_3"),
                vw.stack().to_frame("vwap"),
            ],
            axis=1,
        ).reset_index().rename(columns={"level_0": "trade_time", "level_1": "code"})

        return out.sort_values(["trade_time", "code"]).reset_index(drop=True)

    def concat_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        将 signal_1/2/3 + vwap merge 回原始长表 data（data 必须含 trade_time, code）
        """
        sig = self.get_signals(**kwargs)

        out = data.copy()
        out["trade_time"] = pd.to_datetime(out["trade_time"])

        out = out.merge(
            sig,
            on=["trade_time", "code"],
            how="left",
            sort=False,
        )

        return out.sort_values(["trade_time", "code"]).reset_index(drop=True)

    
class NoiseRangePerpsStrategy(BaseStrategy):
    """
    以分钟 K 线的收盘价突破噪声区域边界作为开仓信号。

    具体地：
    - 当收盘价位于噪声区域内，认为是合理波动，不存在趋势，不产生交易信号；
    - 当收盘价突破噪声区域上边界（UpperBound），认为向上趋势形成，
      发出做多信号，并以下一根 K 线的开盘价开多仓；
    - 当收盘价突破噪声区域下边界（LowerBound），认为向下趋势形成，
      发出做空信号，并以下一根 K 线的开盘价开空仓。

    本策略适用于 Bitcoin 永续合约（24/7 交易）：
    - 不存在固定“收盘”概念；
    - 允许隔夜持仓；
    - 为避免价格在噪声边界附近频繁震荡导致过度交易，
      仅在固定时间间隔内判断是否允许开仓；
    - 为控制风险，一旦在任意时刻触发对向边界，则立即平仓或反手。
    """

    params: Dict = dict(
        commission=0.01,        # 预留交易成本（用于仓位计算的保守折扣）
        hold_num=1,             # 同时持有的合约数量上限（用于资金均分）
        leverage=1,          # 杠杆倍数（永续合约）     # 开仓信号评估间隔（分钟）
        entry_interval=300,      # 开仓信号评估间隔（分钟）
        verbose=False,
    )

    def __init__(self) -> None:
        super().__init__()
        self.order = None
        # 用于控制“仅在固定时间间隔评估开仓信号”
        self._next_entry_time: Dict = {d._name: None for d in self.datas}

    def handle_signal(self, symbol: str) -> None:
        """开仓信号处理"""
        data = self.getdatabyname(symbol)
        size: float = self.getposition(data).size

        if self.signal_1[symbol][0] > self.signal_2[symbol][0]:
            if size < 0:
                self._close_and_reverse(data, f"{symbol} 空头平仓并开多头", self.buy)
            elif size == 0:
                self._open_position(data, f"{symbol} 多头开仓", self.buy)

        elif self.signal_1[symbol][0] < self.signal_3[symbol][0]:
            if size > 0:
                self._close_and_reverse(data, f"{symbol} 多头平仓并开空头", self.sell)
            elif size == 0:
                self._open_position(data, f"{symbol} 空头开仓", self.sell)
        

    def handle_stop_loss(self, symbol: str) -> None:
        """止损 / 反向突破逻辑（每根 K 线检查）"""
        data = self.getdatabyname(symbol)
        size: float = self.getposition(data).size

        if size > 0 and self.signal_1[symbol][0] < self.signal_3[symbol][0]:
            self._close_and_reverse(data, f"{symbol} 多头触发下边界，反手做空", self.sell)

        elif size < 0 and self.signal_1[symbol][0] > self.signal_2[symbol][0]:
            self._close_and_reverse(data, f"{symbol} 空头触发上边界，反手做多", self.buy)

    def handle_take_profit(self, symbol):
        pass

    def _run(self, symbol: str) -> None:

        current_time: str = bt.num2date(
            self.getdatabyname(symbol).datetime[0]
        ).strftime("%H:%M:%S")

        if current_time in ["04:30:00","11:30:00","18:30:00"]:

            self.handle_signal(symbol)

        elif self.getpositionbyname(symbol).size == 0:
            pass

        else:
            self.handle_stop_loss(symbol)
            self.handle_take_profit(symbol)



class BuyAndHoldStrategy(BaseStrategy):
    """
    简单的买入并持有策略（Buy and Hold）

    逻辑：
    - 在回测开始时，以全部资金买入标的资产；
    - 持有至回测结束，不进行任何交易操作。
    """

    params: Dict = dict(
        leverage=1,          # 杠杆倍数（永续合约）
        verbose=False,
    )

    def __init__(self) -> None:
        super().__init__()
        self.order = None
        self._bought: Dict = {d._name: False for d in self.datas}

    def _run(self, symbol: str) -> None:
        """在回测开始时买入，并持有至结束"""
        data = self.getdatabyname(symbol)
        size: float = self.getposition(data).size

        if not self._bought[symbol]:
            self.log(f"{symbol} 买入开仓", verbose=self.p.verbose)
            size = self._calculate_size(data)
            self.order = self.buy(data=data, size=size, exectype=bt.Order.Market)
            self._bought[symbol] = True


    def handle_signal(self, symbol):
        return super().handle_signal(symbol)
    def handle_stop_loss(self, symbol):
        return super().handle_stop_loss(symbol)
    def handle_take_profit(self, symbol):
        return super().handle_take_profit(symbol)
    

if __name__ == "__main__":
    data_path = r".\datasets\tests\test.parquet"
    data = pd.read_parquet(data_path)

    combo_data = FundingNoiseArea(data).fit()
    combo_data.set_index("trade_time", inplace=True)

    
    factors = [col for col in combo_data.columns if col.startswith("signal")]
    factors_value: pd.DataFrame = combo_data[factors].copy()


    print(factors_value.describe().drop("count",axis=0).to_dict())
    result = backtest_strategy(
        data=combo_data,
        code="BTCUSDT",
        strategy=NoiseRangePerpsStrategy,
        strategy_kwargs={"verbose":False,}
    )

    # print("Sharpe Ratio:", get_strategy_sharpe_ratio(result))

    # print("Cumulative Return:", get_strategy_cumulative_return(result).iloc[-  1],"%")
    # print("Max Drawdown:", get_strategy_maxdrawdown(result),"%")
    # print("Win Rate:", get_strategy_win_rate(result).iloc[0]['win_rate']*100,"%")
    # print("Total Commission:", get_strategy_total_commission(result)/COMMISSION["cash"]*100,"%")
    # ax = plot_cumulative_return(result, title="Buy and Hold Strategy")
    # import matplotlib.pyplot as plt
    # plt.savefig("buy_and_hold_cumulative_return.png")