from typing import List, Literal, Optional,Dict, Callable
import numpy as np
import pandas as pd
import talib as ta
import backtrader as bt
from loguru import logger



__all__ = ["BaseSignal","BaseStrategy"]
class BaseSignal:
    """
    Docstring for BaseSignal
    """

    REQUIRED = ("get_signal", "get_factors", "concat_signal")
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [m for m in cls.REQUIRED if m not in cls.__dict__]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define methods: {', '.join(missing)}"
            )
    def __init__(self,ohlcv:pd.DataFrame)->None:
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
            values=["close", "volume","open", "high", "low","amount"],
        ).sort_index()

        self.close: pd.DataFrame = self.pivot_frame["close"]
        self.volume: pd.DataFrame = self.pivot_frame["volume"]
        self.open: pd.DataFrame = self.pivot_frame["open"]
        self.high: pd.DataFrame = self.pivot_frame["high"]
        self.low: pd.DataFrame = self.pivot_frame["low"]
        self.amount: pd.DataFrame = self.pivot_frame["amount"]

    def get_signal(self,**kwargs)->pd.DataFrame:
        """Get signal DataFrame"""
        pass

    def get_factors(self,factor_name:Literal["factor1","factor2"],**kwargs)->pd.DataFrame:
        """Get factor DataFrame"""
        pass


    def concat_signal(self,data:pd.DataFrame)->pd.DataFrame:
        """Concatenate signal to original data"""
        pass


    def fit(self) -> pd.DataFrame:
        return self.concat_signal(self.ohlcv)

    def calculate_rolling_vwap(
        self,
        window: int,
    ) -> pd.Series:
        """
        Calculate rolling VWAP (no daily reset, perps-friendly)

        Parameters
        ----------
        df : pd.DataFrame
            Time-indexed OHLCV data
        window : int
            Rolling window length (number of bars)
        price_col : str
            Price column used in VWAP
        volume_col : str
            Volume column used in VWAP

        Returns
        -------
        pd.Series
            Rolling VWAP aligned with df.index
        """

        pv = self.close * self.volume

        vwap = (
            pv.rolling(window=window, min_periods=1).sum()
            / self.volume.rolling(window=window, min_periods=1).sum()
        )

        return vwap
    


class BaseStrategy(bt.Strategy):

    """BaseStratgy template for backtesting strategies."""
    params: Dict = dict(
        commission=0.01,
        hold_num=1,
        leverage=1,
        verbose=False
    ) # 预留1%的交易成本
    REQUIRED = ("handle_signal", "handle_stop_loss", "handle_take_profit")

    def __init__(self) -> None:
        
        self.order = None
        self.signal: Dict = {d._name: d.signal for d in self.datas}
        self.factor1: Dict = {d._name: d.factor1 for d in self.datas}
        self.factor2: Dict = {d._name: d.factor2 for d in self.datas}
        self.log(f"策略初始化完成 - commission: {self.p.commission}",pd.Timestamp.now(), verbose=self.p.verbose)
        

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [m for m in cls.REQUIRED if m not in cls.__dict__]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define methods: {', '.join(missing)}"
            )

    def handle_signal(self, symbol: str) -> None:
        raise NotImplementedError

    def handle_stop_loss(self, symbol: str) -> None:
        raise NotImplementedError

    def handle_take_profit(self, symbol: str) -> None:
        raise NotImplementedError

    def log(self, msg: str, current_dt: pd.Timestamp = None, verbose: bool = False):
        if current_dt is None:
            current_dt: pd.Timestamp = self.datetime.datetime(0)
        if verbose:
            logger.info(f"{current_dt} {msg}")


    def _calculate_size(self, data) -> float:
        """
        Perps position calculation:
        - size = notional / price
        - introducing leverage into the system
        """
        price = data.close[0]
        equity = self.broker.getvalue()

        alloc_cash = equity * (1 - self.p.commission) / self.p.hold_num
        notional = alloc_cash * self.p.leverage

        return notional / price if price > 0 else 0.0

    def _close_and_reverse(self, data, reason: str, new_action: Callable) -> None:
        self.log(reason, verbose=self.p.verbose)
        self.order = self.close(data=data)
        size = self._calculate_size(data)
        self.order = new_action(data=data, size=size, exectype=bt.Order.Market)

    def _open_position(self, data, reason: str, action: Callable) -> None:
        self.log(reason, verbose=self.p.verbose)
        size = self._calculate_size(data)
        self.order = action(data=data, size=size, exectype=bt.Order.Market)

    def next(self) -> None:

        for data in self.datas:

            if self.datetime.datetime(0) != data.datetime.datetime(0):
                continue

            if self.order:
                self.cancel(self.order)
                self.order = None

            self._run(data._name)

    def prenext(self) -> None:
        self.next()



    def _run(self, symbol: str) -> None:
        """Trade every bar based on signal, stop loss and take profit."""
        current_time: str = bt.num2date(
            self.getdatabyname(symbol).datetime[0]
        ).strftime("%H:%M:%S")
        self.handle_stop_loss(symbol)
        self.handle_take_profit(symbol)
        self.handle_signal(symbol)



    def rebalance(self, symbol: str) -> None:
        """Clear all positions based on signal"""
        if self.getpositionbyname(symbol).size != 0:
            self.order = self.close(data=symbol, exectype=bt.Order.Market)
            self.log(f"{symbol} 平仓", verbose=self.p.verbose)