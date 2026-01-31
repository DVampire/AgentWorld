from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)
from src.environment.quickbacktest.backtest import backtest_strategy,STRATEGY_PARAMS,COMMISSION
from src.environment.quickbacktest.utils import (
    get_strategy_sharpe_ratio,
    get_strategy_cumulative_return,
    get_strategy_maxdrawdown,
    get_strategy_total_commission,
    get_strategy_win_rate,
    plot_cumulative_return
)
from src.environment.quickbacktest.base_types import BaseStrategy,BaseSignal
import pandas as pd
from typing import Literal, Tuple, Union, List, Dict, Any
import numpy as np
import backtrader as bt



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
    ----------------------------

    以 funding 周期替代“天”，构造噪声区域：

    1) 将时间按 funding 周期划分；
    2) 每个周期的第一根 bar 的 close 作为锚点 A_k；
    3) 在相同的周期内相对位置 j 上，用历史周期的位移分布估计 sigma；
    4) UpperBound = A_k * (1 + sigma)，LowerBound = A_k * (1 - sigma)；
    5) 计算周期内VWAP，并拼接 signal=close。

    Attributes:
        ohlcv (pd.DataFrame): OHLCV数据（长表）
        pivot_frame (pd.DataFrame): 透视表（分钟索引、code列）
        close (pd.DataFrame): 收盘价数据
        volume (pd.DataFrame): 成交量数据

    Methods:
        calculate_cycle_id(funding_hours: int = 8, tz: str = "UTC") -> pd.DataFrame:
            计算 funding 周期编号与周期内相对位置

        calculate_funding_anchor() -> pd.DataFrame:
            计算每个 funding 周期的锚点价格 A_k

        calculate_funding_vwap() -> pd.DataFrame:
            计算 funding 周期内的累计VWAP

        calculate_sigma(window: int = 14, mode: str = "mean", q: float = 0.85, use_shift: bool = True) -> pd.DataFrame:
            计算 sigma（按相对位置j分组的滚动统计），默认shift避免同周期污染

        calculate_bound(window: int = 14, method: str = "U", **kwargs) -> pd.DataFrame:
            计算上下边界

        concat_signal(data: pd.DataFrame, window: int = 14, **kwargs) -> pd.DataFrame:
            拼接 upperbound/lowerbound/vwap/signal（signal为close）

        fit(window: int = 14, **kwargs) -> pd.DataFrame:
            输出结果
    """

    def __init__(self, ohlcv: pd.DataFrame) -> None:
        """
        初始化FundingNoiseArea对象。

        参数:
            ohlcv (pd.DataFrame): 包含 code, trade_time, close, volume 字段的DataFrame。
        """
        required = {"code", "trade_time", "close", "volume"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"ohlcv is missing required columns: {sorted(missing)}")

        self.ohlcv: pd.DataFrame = ohlcv.copy()
        self.ohlcv["trade_time"] = pd.to_datetime(self.ohlcv["trade_time"])

        self.pivot_frame: pd.DataFrame = pd.pivot_table(
            self.ohlcv,
            index="trade_time",
            columns="code",
            values=["close", "volume"],
        ).sort_index()

        self.close: pd.DataFrame = self.pivot_frame["close"]
        self.volume: pd.DataFrame = self.pivot_frame["volume"]


    def _cycle_id_pos(self, funding_hours: int = 8, tz: str = "Asia/Shanghai") -> tuple[np.ndarray, np.ndarray]:
        """
        计算东八区对齐的 funding 周期 CycleID 与周期内位置 pos_in_cycle

        规则：
            - 以东八区本地日 00:00 为起点
            - 每 funding_hours 小时一个周期（默认8小时）
            - 周期边界为 00:00 / 08:00 / 16:00

        返回:
            cycle_id (np.ndarray): 每个trade_time对应的全局周期ID（长度=len(index)）
            pos_in_cycle (np.ndarray): 每个trade_time在其周期内的序号（从0递增）
        """
        idx = self.close.index

        # 保证是 tz-aware 再转东八区
        ts = idx
        if ts.tz is None:
            ts = ts.tz_localize("UTC")  # 若你的trade_time本来就是东八区，请改成 tz_localize("Asia/Shanghai")
        ts8 = ts.tz_convert(tz)

        # 当天零点（东八区）
        day0 = ts8.normalize()

        # 距离零点的秒数
        sec_from_day0 = (ts8.view("int64") - day0.view("int64")) // 10**9
        funding_seconds = funding_hours * 3600

        cycle_in_day = (sec_from_day0 // funding_seconds).astype(np.int64)  # 0,1,2
        day_index = day0.date.astype("datetime64[D]").astype(np.int64)       # 跨天递增的整数

        cycle_id = day_index * (24 // funding_hours) + cycle_in_day          # 全局周期ID

        # 周期内位置：对同一cycle_id做cumcount
        pos_in_cycle = pd.Series(cycle_id, index=idx).groupby(cycle_id).cumcount().values.astype(np.int64)

        return cycle_id, pos_in_cycle


    def calculate_cycle_id(self, funding_hours: int = 8, tz: str = "UTC") -> pd.DataFrame:
        """
        计算 funding 周期编号与周期内相对位置（以bar计）

        参数:
            funding_hours (int): funding 周期长度（小时）
            tz (str): 对齐时区（建议UTC）

        返回:
            pd.DataFrame: 包含 cycle_id 与 pos_in_cycle（与 close 同shape，index=trade_time, columns=code）
        """
        idx = self.close.index
        ts = idx.tz_localize(None)

        # 以 epoch 为对齐基准：cycle_id = floor(timestamp / funding_seconds)
        funding_seconds = funding_hours * 3600
        seconds = (ts.view("int64") // 10**9).astype(np.int64)
        cycle_id = seconds // funding_seconds

        # 周期内位置（bar序号）需要按“每周期的第一根bar”为0递增
        # 这里先用 cycle_id 作为分组键，之后用 groupby/cumcount 得到位置
        cycle_df = pd.DataFrame({"cycle_id": cycle_id}, index=idx)
        pos = cycle_df.groupby("cycle_id").cumcount()

        # broadcast 到所有 code 列
        cycle_id_df = pd.DataFrame(
            np.repeat(cycle_id[:, None], self.close.shape[1], axis=1),
            index=idx,
            columns=self.close.columns,
        )
        pos_df = pd.DataFrame(
            np.repeat(pos.values[:, None], self.close.shape[1], axis=1),
            index=idx,
            columns=self.close.columns,
        )

        return pd.concat(
            [cycle_id_df.stack().rename("cycle_id"), pos_df.stack().rename("pos_in_cycle")],
            axis=1,
        ).reset_index().rename(columns={"level_0": "trade_time", "level_1": "code"})
    
    def calculate_sigma(
        self,
        window: int = 14,
        mode: Literal["mean", "quantile"] = "mean",
        q: float = 0.85,
        funding_hours: int = 8,
        use_shift: bool = True,
    ) -> pd.DataFrame:
        """
        计算 sigma（Funding-NoiseArea）

        逻辑：
        - anchor A_k：周期第一根 close
        - move_t = |close / anchor - 1|
        - 对相同 pos_in_cycle 的 move 做滚动统计（跨周期），得到 sigma
        - 可选 use_shift=True：将 sigma shift(1) 避免同周期污染

        参数:
            window (int): 滚动窗口（以“周期数”计的近似。这里按 pos 分组后的序列长度滚动）
            mode (str): "mean" 或 "quantile"
            q (float): 分位数（mode="quantile"使用）
            funding_hours (int): funding 周期长度（小时）
            use_shift (bool): 是否对sigma做 shift(1)

        返回:
            pd.DataFrame: index=trade_time, columns=code
        """
        anchor = self.calculate_funding_anchor(funding_hours=funding_hours)
        move = (self.close.div(anchor) - 1.0).abs()

        # 构造 pos_in_cycle
        idx = self.close.index
        ts = idx.tz_localize(None)
        funding_seconds = funding_hours * 3600
        seconds = (ts.view("int64") // 10**9).astype(np.int64)
        cycle_id, pos_in_cycle = self._cycle_id_pos(funding_hours=funding_hours, tz="Asia/Shanghai")
        pos_in_cycle = pd.Series(cycle_id, index=idx).groupby(cycle_id).cumcount()

        sigma = pd.DataFrame(index=idx, columns=self.close.columns, dtype=float)

        # 按 code、pos_in_cycle 分组滚动
        for c in self.close.columns:
            s = move[c].copy()
            df = pd.DataFrame({"move": s.values, "pos": pos_in_cycle.values}, index=idx)

            if mode == "mean":
                out = df.groupby("pos", group_keys=False)["move"].apply(
                    lambda x: x.rolling(window=window, min_periods=window).mean()
                )
            elif mode == "quantile":
                out = df.groupby("pos", group_keys=False)["move"].apply(
                    lambda x: x.rolling(window=window, min_periods=window).quantile(q)
                )
            else:
                raise ValueError("mode must be 'mean' or 'quantile'")

            if use_shift:
                out = df.groupby("pos", group_keys=False)["move"].apply(
                    lambda x: (
                        x.rolling(window=window, min_periods=window).mean()
                        if mode == "mean"
                        else x.rolling(window=window, min_periods=window).quantile(q)
                    ).shift(1)
                )

            sigma[c] = out.values

        return sigma

    def calculate_funding_anchor(self, funding_hours: int = 8) -> pd.DataFrame:
        """
        计算每个 funding 周期的锚点价格 A_k（周期第一根bar的close）

        参数:
            funding_hours (int): funding 周期长度（小时）

        返回:
            pd.DataFrame: index=trade_time, columns=code
        """
        idx = self.close.index
        ts = idx.tz_localize(None)
        funding_seconds = funding_hours * 3600
        seconds = (ts.view("int64") // 10**9).astype(np.int64)
        cycle_id, _ = self._cycle_id_pos(funding_hours=funding_hours, tz="Asia/Shanghai")

        # 对每个 code：按 cycle_id 分组取第一根 close，并 forward-fill 到周期内所有bar
        anchor = self.close.copy()
        for c in anchor.columns:
            s = anchor[c]
            a = s.groupby(cycle_id).transform("first")
            anchor[c] = a
        return anchor

    def get_signal(self)->pd.DataFrame:
        """Get signal DataFrame"""
        
        return (
            self.ohlcv.set_index(["trade_time", "code"])["close"]
            .to_frame(name="signal")
        )
    
    def get_factors(self,window,mode,q,funding_hours,use_shift,factor_name:Literal["factor1","factor2"])->pd.DataFrame:
        """Get factor DataFrame"""
        if factor_name == "factor1":
            return self.calculate_bound(
            window=window,
            method="U",
            mode=mode,
            q=q,
            funding_hours=funding_hours,
            use_shift=use_shift,
        )
        elif factor_name == "factor2":
            return self.calculate_bound(
            window=window,
            method="L",
            mode=mode,
            q=q,
            funding_hours=funding_hours,
            use_shift=use_shift,
        )
        else:
            raise ValueError("factor_name must be 'factor1' or 'factor2'")
        

    def calculate_bound(
        self,
        window: int = 14,
        method: str = "U",
        mode: Literal["mean", "quantile"] = "mean",
        q: float = 0.85,
        funding_hours: int = 8,
        use_shift: bool = True,
    ) -> pd.DataFrame:
        """计算上下边界（Funding-NoiseArea）"""
        anchor = self.calculate_funding_anchor(funding_hours=funding_hours)
        sigma = self.calculate_sigma(
            window=window,
            q=q,
            mode=mode,
            funding_hours=funding_hours,
            use_shift=use_shift,
        )

        if method.upper() == "U":
            return anchor.mul(1.0 + sigma)
        elif method.upper() == "L":
            return anchor.mul(1.0 - sigma)
        else:
            raise ValueError("method must be 'U' or 'L'")

    def concat_signal(
        self,
        data: pd.DataFrame,
        window: int = 30,
        mode: Literal["mean", "quantile"] = "quantile",
        q: float = 0.85,
        funding_hours: int = 8,
        use_shift: bool = True,
    ) -> pd.DataFrame:
        """
        拼接噪声区域结果（signal为close）

        返回:
            pd.DataFrame: 包含 upperbound, lowerbound, vwap, signal 的长表
        """
        upperbound: pd.DataFrame = self.get_factors(
            window=window,
            factor_name="factor1",
            mode=mode,
            q=q,
            funding_hours=funding_hours,
            use_shift=use_shift,
        )
        lowerbound: pd.DataFrame = self.get_factors(
            window=window,
            factor_name="factor2",
            mode=mode,
            q=q,
            funding_hours=funding_hours,
            use_shift=use_shift,
        )
        vwaps: pd.DataFrame = self.calculate_rolling_vwap(window=window)

        signal: pd.DataFrame = self.get_signal()

        return (
            pd.concat(
                [
                    data.set_index(["trade_time", "code"]),
                    upperbound.stack().to_frame(name="factor1"),
                    signal,
                    lowerbound.stack().to_frame(name="factor2"),
                    vwaps.stack().to_frame(name="vwap"),
                ],
                axis=1,
            )
            .reset_index()
            .sort_values(["trade_time", "code"])
            .reset_index(drop=True)
        )

    def fit(
        self,
        window: int = 21,
        mode: Literal["mean", "quantile"] = "mean",
        q: float = 0.85,
        funding_hours: int = 8,
        use_shift: bool = True,
    ) -> pd.DataFrame:
        """生成最终结果表"""
        return self.concat_signal(
            self.ohlcv,
            window=window,
            mode=mode,
            q=q,
            funding_hours=funding_hours,
            use_shift=use_shift,
        )

    
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

        # 信号与边界（由 datafeed 提供）
        self.signal: Dict = {d._name: d.signal for d in self.datas}
        self.factor1: Dict = {d._name: d.factor1 for d in self.datas}
        self.factor2: Dict = {d._name: d.factor2 for d in self.datas}

        # 用于控制“仅在固定时间间隔评估开仓信号”
        self._next_entry_time: Dict = {d._name: None for d in self.datas}

    def handle_signal(self, symbol: str) -> None:
        """开仓信号处理"""
        data = self.getdatabyname(symbol)
        size: float = self.getposition(data).size

        if self.signal[symbol][0] > self.factor1[symbol][0]:
            if size < 0:
                self._close_and_reverse(data, f"{symbol} 空头平仓并开多头", self.buy)
            elif size == 0:
                self._open_position(data, f"{symbol} 多头开仓", self.buy)

        elif self.signal[symbol][0] < self.factor2[symbol][0]:
            if size > 0:
                self._close_and_reverse(data, f"{symbol} 多头平仓并开空头", self.sell)
            elif size == 0:
                self._open_position(data, f"{symbol} 空头开仓", self.sell)
        

    def handle_stop_loss(self, symbol: str) -> None:
        """止损 / 反向突破逻辑（每根 K 线检查）"""
        data = self.getdatabyname(symbol)
        size: float = self.getposition(data).size

        if size > 0 and self.signal[symbol][0] < self.factor2[symbol][0]:
            self._close_and_reverse(data, f"{symbol} 多头触发下边界，反手做空", self.sell)

        elif size < 0 and self.signal[symbol][0] > self.factor1[symbol][0]:
            self._close_and_reverse(data, f"{symbol} 空头触发上边界，反手做多", self.buy)

    def handle_take_profit(self, symbol):
        pass

    def _run(self, symbol: str) -> None:

        current_time: str = bt.num2date(
            self.getdatabyname(symbol).datetime[0]
        ).strftime("%H:%M:%S")

        if current_time in ["04:30:00","11:30:00","18:30:00"]:

            self.handle_signal(symbol)

        elif current_time in ["23:55:00","07:55:00","15:55:00"]:
            self.rebalance(symbol)

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
    result = backtest_strategy(
        data=combo_data,
        code="BTCUSDT",
        strategy=NoiseRangePerpsStrategy,
        strategy_kwargs={"verbose":False,}
    )

    print("Sharpe Ratio:", get_strategy_sharpe_ratio(result))

    print("Cumulative Return:", get_strategy_cumulative_return(result).iloc[-  1],"%")
    print("Max Drawdown:", get_strategy_maxdrawdown(result),"%")
    print("Win Rate:", get_strategy_win_rate(result).iloc[0]['win_rate']*100,"%")
    print("Total Commission:", get_strategy_total_commission(result)/COMMISSION["cash"]*100,"%")
    ax = plot_cumulative_return(result, title="Buy and Hold Strategy")
    import matplotlib.pyplot as plt
    plt.savefig("buy_and_hold_cumulative_return.png")