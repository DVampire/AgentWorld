import pandas as pd
import talib
from copy import deepcopy

from src.indicators.base import Indicator
from src.registry import INDICATOR

@INDICATOR.register_module(force=True)
class OBV(Indicator):
    """
    On-Balance Volume (OBV) indicator.
    """
    def __init__(self, **kwargs):
        super(OBV, self).__init__()
        self.indicators_name = ["obv"]
        
    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call the OBV indicator with the given arguments.
        """
        df = deepcopy(df)
        
        obv = talib.OBV(df["close"], df["volume"])
        
        df["obv"] = obv
        
        res = df[["obv"]]
        return res