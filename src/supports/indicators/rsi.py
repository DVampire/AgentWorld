import pandas as pd
import talib
from copy import deepcopy

from src.supports.indicators.base import Indicator
from src.supports.registry import INDICATOR

@INDICATOR.register_module(force=True)
class RSI(Indicator):
    """
    Relative Strength Index (RSI) indicator.
    """
    def __init__(self, **kwargs):
        super(RSI, self).__init__()
        self.indicators_name = ["rsi"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call the RSI indicator with the given arguments.
        """
        df = deepcopy(df)
        
        rsi = talib.RSI(df["close"], timeperiod=14)
        
        df["rsi"] = rsi
        
        res = df[["rsi"]]
        
        return res