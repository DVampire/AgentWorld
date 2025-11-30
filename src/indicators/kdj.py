import pandas as pd
import talib
from copy import deepcopy

from src.indicators.base import Indicator
from src.registry import INDICATOR

@INDICATOR.register_module(force=True)
class KDJ(Indicator):
    """
    KDJ indicator.
    """
    def __init__(self, **kwargs):
        super(KDJ, self).__init__()
        self.indicators_name = ["stoch_k", "stoch_d"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call the KDJ indicator with the given arguments.
        """
        df = deepcopy(df)
        
        stoch_k, stoch_d = talib.STOCH(df["high"], df["low"], df["close"], fastk_period=14, slowk_period=3, slowd_period=3)
        
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d
        
        res = df[["stoch_k", "stoch_d"]]
        
        return res