import pandas as pd
import talib
from copy import deepcopy

from src.supports.indicators.base import Indicator
from src.supports.registry import INDICATOR

@INDICATOR.register_module(force=True)
class SMA(Indicator):
    """
    Simple Moving Average (SMA) indicator.
    """
    def __init__(self, **kwargs):
        super(SMA, self).__init__()
        self.indicators_name = ["sma_20", "sma_50"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call the SMA indicator with the given arguments.
        """
        df = deepcopy(df)
        # sma_20
        df["sma_20"] = talib.SMA(df["close"], timeperiod=20)
        # sma_50
        df["sma_50"] = talib.SMA(df["close"], timeperiod=50)
        
        res = df[["sma_20", "sma_50"]]
        
        return res