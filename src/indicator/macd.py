import pandas as pd
import talib
from copy import deepcopy

from src.indicator.types import Indicator
from src.registry import INDICATOR

@INDICATOR.register_module(force=True)
class MACD(Indicator):
    """
    Moving Average Convergence Divergence (MACD) indicator.
    """
    def __init__(self, **kwargs):
        super(MACD, self).__init__()
        self.indicators_name = ["macd", "macd_signal", "macd_hist"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call the MACD indicator with the given arguments.
        """
        df = deepcopy(df)
        
        macd, macd_signal, macd_hist = talib.MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)
        
        df["macd"] = macd
        df["macd_signal"] = macd_signal
        df["macd_hist"] = macd_hist
        
        res = df[["macd", "macd_signal", "macd_hist"]]
        
        return res