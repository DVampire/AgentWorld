import pandas as pd
import talib
from copy import deepcopy

from src.supports.indicators.base import Indicator
from src.supports.registry import INDICATOR

@INDICATOR.register_module(force=True)
class MFI(Indicator):
    """
    Money Flow Index (MFI) indicator.
    """
    def __init__(self, **kwargs):
        super(MFI, self).__init__()
        self.indicators_name = ["mfi"]
        
    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call the MFI indicator with the given arguments.
        """
        df = deepcopy(df)
        
        mfi = talib.MFI(df["high"], df["low"], df["close"], df["volume"], timeperiod=14)
        
        df["mfi"] = mfi
        
        res = df[["mfi"]]
        return res