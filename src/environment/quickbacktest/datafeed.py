from typing import List, Tuple

import backtrader as bt
from backtrader.feeds import PandasDirectData

__all__ = ["CryptoDataFeed"]

class CryptoDataFeed(PandasDirectData):
    """
    OHLC 为后复权

    datetime必须为datetime64[ns]类型，其他字段不支int,float以外类型
    """

    params: Tuple[Tuple] = (
        ("datetime", 0),
        ("open", 1),
        ("high", 2),
        ("low", 3),
        ("close", 4),
        ("volume", 5),
        ("amount", 6),
        ("signal",7),
        ("factor1",8),
        ("factor2",9),
        ("vwap",10), # vwap
        ("dtformat","%Y-%m-%d %H:%M:%S"),
        ("timeframe",bt.TimeFrame.Minutes),
    )

    lines: List[str] = (
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "signal",
        "factor1",
        "factor2",
        "vwap",        
    )