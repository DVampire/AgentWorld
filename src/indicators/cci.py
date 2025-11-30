import pandas as pd
import talib
from copy import deepcopy

from src.indicators.base import Indicator
from src.registry import INDICATOR

@INDICATOR.register_module(force=True)
class CCI(Indicator):
    """
    Commodity Channel Index (CCI) indicator.
    """
    def __init__(self, **kwargs):
        super(CCI, self).__init__()
        self.indicators_name = ["cci"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call the CCI indicator with the given arguments.
        """
        df = deepcopy(df)
        
        cci = talib.CCI(df["high"], df["low"], df["close"], timeperiod=14)
        
        df["cci"] = cci
        
        res = df[["cci"]]
        
        return res