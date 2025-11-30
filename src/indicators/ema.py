import pandas as pd
import talib
from copy import deepcopy

from src.indicators.base import Indicator
from src.registry import INDICATOR

@INDICATOR.register_module(force=True)
class EMA(Indicator):
    """
    Exponential Moving Average (EMA) indicator.
    """
    def __init__(self, **kwargs):
        super(EMA, self).__init__()
        self.indicators_name = ["ema_20", "ema_50"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call the EMA indicator with the given arguments.
        """
        df = deepcopy(df)
        # ema_20 
        df["ema_20"] = talib.EMA(df["close"], timeperiod=20)
        # ema_50
        df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
        
        res = df[["ema_20", "ema_50"]]
        
        return res