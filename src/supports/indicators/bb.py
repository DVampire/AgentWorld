import pandas as pd
import talib
from copy import deepcopy

from src.supports.indicators.base import Indicator
from src.supports.registry import INDICATOR

@INDICATOR.register_module(force=True)
class BB(Indicator):
    """
    Bollinger Bands (BB) indicator.
    """
    def __init__(self, **kwargs):
        super(BB, self).__init__()
        self.indicators_name = ["bb_upper", "bb_middle", "bb_lower"]

    async def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Call the BB indicator with the given arguments.
        """
        df = deepcopy(df)
        
        bb_upper, bb_middle, bb_lower = talib.BBANDS(df["close"], timeperiod=20, nbdevup=2, nbdevdn=2)
        
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower
        
        res = df[["bb_upper", "bb_middle", "bb_lower"]]
        
        return res