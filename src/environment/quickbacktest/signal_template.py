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
    Avoid look-head bias when generating signals and factors.
    When coding, always use tz-aware DatetimeIndex.

    Keep the class name same as module name for dynamic loading.

    Example: module name: MySignal  -> class name: MySignal

    Write docstrings for the class at here. Follow the format below (to describe the meaning of each singnal and factor)
    Do not include anything else here except the dictionary style docstring, follow strick json format.

    Any format information about signal and factor generation must be in folling methods, like get_signal, get_factors, concat_signal,etc.

    Leave range number all be -1 when generating the docstring.

    Update and Add module will trigger getSignalQuantile tool to update the range automatically and return the updated range information as output.

    You can also use getdocstring tool to get the docstring after updating the range.

    Do not update range by yourself.

    


                    {
                        "signal":{
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
                                            "hit_rate": float
                                        }
                        },
                        "factor1":{
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
                                        "hit_rate": float
                                    }

                        },
                        "factor2":{
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
                                        "hit_rate": float
                                    }
                            },

                        "why_these_factors": string
                    }

                    
    Outputs consumed by Strategy
    ----------------------------
    - signal   : primary decision input (price / score / indicator) - can be understood as a factor with high IC value with returns
    - factor1  : free-form auxiliary factor
    - factor2  : free-form auxiliary factor

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

    def get_signal(self, **kwargs) -> pd.DataFrame:
        """Generate signal (main strategy input)."""
        pass

    def get_factors(
        self,
        factor_name: Literal["factor1", "factor2"],
        **kwargs
    ) -> pd.DataFrame:
        """Generate factor1 or factor2 (free-form auxiliary factors)."""
        pass

    def concat_signal(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Attach signal / factor1 / factor2 back to the original OHLCV data.

        What this method does (trading pipeline view)
        ---------------------------------------------
        - Convert wide matrices (signal / factors) into long format
        - Align everything by (trade_time, code)
        - Return a long table that can be fed into Backtrader datafeeds

        Required output columns
        -----------------------
        - trade_time
        - code
        - signal
        - factor1
        - factor2
        - vwap

        vwaps: pd.DataFrame = self.calculate_rolling_vwap(window:int) # ALREADY IMPLEMENTED IN BaseSignal

        Example structure (illustrative only)
        -------------------------------------
        # signal (wide)   -> index=trade_time, columns=code
        # factor1 (wide)  -> index=trade_time, columns=code
        # factor2 (wide)  -> index=trade_time, columns=code

        # Convert to long and merge:
        #
        # signal_long  = signal.stack().to_frame("signal")
        # factor1_long = factor1.stack().to_frame("factor1")
        # factor2_long = factor2.stack().to_frame("factor2")
        #
        # out = (
        #   data.set_index(["trade_time", "code"])
        #       .join([signal_long, factor1_long, factor2_long])
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