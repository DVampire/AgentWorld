"""AgentSignal Template"""

from src.environment.quickbacktest.base_types import BaseSignal
from typing import Literal
import pandas as pd
import talib as ta


class AgentSignal(BaseSignal):
    """
    AgentSignal
    ===========

    This class prepares trading inputs for the strategy.
    It does NOT execute trades.
    Avoid look-head bias when generating signals.
    When coding, always use tz-aware DatetimeIndex.

    Keep the class name same as module name for dynamic loading.

    Example: module name: MySignal  -> class name: MySignal

    Write docstrings for the class at here. Follow the format below (to describe the meaning of each singnal and factor)
    Do not include anything else here except the dictionary style docstring, follow strick json format.

    Leave range number all be -1 when generating the docstring.

    Update and Add module will trigger getSignalQuantile tool to update the range automatically and return the updated range information as output.

    You can also use getdocstring tool to get the docstring after updating the range.

    Do not update range by yourself.

    Name follow convention {key_variable}_{version}, e.g. momentum_v1, volatility_v1, etc.
    


                    {
                        "signal_1":{
                                    "name":string
                                    "explanation": string
                                    "range": {
                                            "mean": float,
                                            "std": float,
                                            "min": float,
                                            "25%": float,
                                            "50%": float,
                                            "75%": float ,
                                            "max": float,
                                        }
                        },
                        "signal_2":{
                                    "name":string,
                                    "explanation":string
                                    "range": {
                                        "mean": float,
                                        "std": float,
                                        "min": float,
                                        "25%": float,
                                        "50%": float,
                                        "75%": float ,
                                        "max": float,
                                    }

                        },
                        "signal_3":{
                                    "name":string,
                                    "explanation":string
                                    "range": {
                                        "mean": float,
                                        "std": float,
                                        "min": float,
                                        "25%": float,
                                        "50%": float,
                                        "75%": float ,
                                        "max": float,
                                    }
                            },

                        "why_these_signals": string
                    }

                    
    Outputs consumed by Strategy
    ----------------------------
    - signal_1   : primary decision input (price / score / indicator) - can be understood as a factor with high IC value with returns
    - signal_2   : secondary decision input (price / score / indicator) - can be understood as a factor with high IC value with returns
    - signal_3   : tertiary decision input (price / score / indicator) - can be understood as a factor with high IC value with returns

    Data has been initlized in BaseSignal in the format:

    ...
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

    You are direcly access using self.close, self.volume, etc.

    """

    def get_signals(self, **kwargs) -> pd.DataFrame:
        """Generate signal DataFrame with columns: signal_1, signal_2, signal_3"""
        pass


    def concat_signals(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Attach signal_1 / signal_2 / signal_3 back to the original OHLCV data.

        What this method does (trading pipeline view)
        ---------------------------------------------
        - Convert wide matrices (signal_1 / signal_2 / signal_3) into long format
        - Align everything by (trade_time, code)
        - Return a long table that can be fed into Backtrader datafeeds

        Required output columns
        -----------------------
        - trade_time
        - code
        - signal_1
        - signal_2
        - signal_3
        - vwap

        if you need resample the data into highr level, e.g. from 1m to 1D, please make sure to use .shift(1) to prevent look-ahead bias.
        example:
            close_s = self.close[code].resample('1D').last().shift(1)
            high_s = self.high[code].resample('1D').max().shift(1)
            low_s = self.low[code].resample('1D').min().shift(1)
        MUST DO IT THIS WAY!!!

        vwaps: pd.DataFrame = self.calculate_rolling_vwap(window:int) # ALREADY IMPLEMENTED IN BaseSignal

        Example structure (illustrative only)
        -------------------------------------
        # signal_1(wide)  -> index=trade_time, columns=code
        # signal_2(wide)  -> index=trade_time, columns=code
        # signal_3(wide)  -> index=trade_time, columns=code

        # Convert to long and merge:
        #
        # signal_1_long  = signal_1.stack().to_frame("signal_1")
        # signal_2_long = signal_2.stack().to_frame("signal_2")
        # signal_3_long = signal_3.stack().to_frame("signal_3")
        #
        # out = (
        #   data.set_index(["trade_time", "code"])
        #       .join([signal_1_long, signal_2_long, signal_3_long])
        #       .reset_index()
        #       .sort_values(["trade_time", "code"])
        # )

        Rules
        -----
        - kwargs must be read via kwargs.get("x", default)
        - kwargs may be empty
        - Do not drop or reorder original rows
        - Do not interpret trading logic here
        """
        pass