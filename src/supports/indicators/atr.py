import pandas as pd
import talib
from copy import deepcopy

from src.supports.indicators.base import Indicator
from src.supports.registry import INDICATOR

@INDICATOR.register_module(force=True)
class ATR(Indicator):
    """
    Average True Range (ATR) indicator.
    """
    def __init__(self, **kwargs):
        super(ATR, self).__init__()
        self.indicators_name = ["atr"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call the ATR indicator with the given arguments.
        """
        df = deepcopy(df)
        
        atr = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14)
        
        df["atr"] = atr
        
        res = df[["atr"]]
        
        return res