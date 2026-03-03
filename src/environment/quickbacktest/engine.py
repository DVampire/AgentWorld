import backtrader as bt
import pandas as pd
from loguru import logger
from tqdm.notebook import tqdm as track
from typing import Tuple, List
__all__ = ["BackTesting", "PerpCommission", "check_dataframe_cols"]

# 永续合约手续费与保证金模型
class PerpCommission(bt.CommInfoBase):
    """
    Bitcoin 永续合约费用模型（future-like）：
    - 双边手续费（开仓/平仓均收）
    - 使用保证金交易
    - 逐 bar 盯市（PnL 实时计入 cash）
    """
    params = (
        ("commission", 0.0004),          # 手续费比例（如 taker 0.04% = 0.0004）
        ("commtype", bt.CommInfoBase.COMM_PERC),
        ("percabs", True),
        ("stocklike", False),            # 永续合约必须为 False
        ("margin", 1.0),                 # 初始保证金率 = 1 / leverage
        ("mult", 1.0),                   # 合约乘数（USDT 本位线性永续通常为 1.0）
    )

    def _getcommission(self, size, price, pseudoexec):
        return abs(size) * price * self.p.commission


# DataFrame 列检查与对齐
def check_dataframe_cols(
    dataframe: pd.DataFrame, datafeed_cls: bt.feeds.PandasData
) -> bool:
    """
    检查数据框的列。

    参数:
        dataframe (pd.DataFrame): 要检查的数据框。
        datafeed_cls (bt.feeds.PandasData): 数据源类。

    返回值:
        bool: 如果检查通过，则返回True；否则返回False。
    """
    if not isinstance(datafeed_cls, bt.feed.MetaAbstractDataBase):
        raise ValueError(
            "datafeed_cls must be a subclass of bt.feeds.PandasData or bt.feeds.PandasDirectData"
        )

    cols: List[Tuple] = [
        (k, v)
        for k, v in datafeed_cls.params.__dict__.items()
        if isinstance(v, int) and k not in ("timeframe", "dtformat")
    ]
    sorted_cols: List[Tuple] = sorted(cols, key=lambda x: x[1])
    return dataframe[
        [
            v[0]
            for v in sorted_cols
            if (v[0] != "datetime") and (v[1] != 0 or v[1] is not None)
        ]
    ]


class BackTesting:

    def __init__(
        self,
        cash: float,
        commission: float = 0.0004,
        slippage_perc: float = 0.0001,
        leverage: float = 10.0,
    ) -> None:

        self.cerebro = bt.Cerebro()

        # 永续合约手续费与保证金
        comminfo = PerpCommission(
            commission=commission,
            margin=1.0 / leverage,
            mult=1.0,
        )
        self.cerebro.broker.addcommissioninfo(comminfo)

        # 滑点（双边）
        self.cerebro.broker.set_slippage_perc(perc=slippage_perc)

        # 初始资金
        self.cerebro.broker.set_cash(cash)


        self.cerebro.broker.set_shortcash(True)

        # 存储原始数据
        self.datas = pd.DataFrame()

    def load_data(
        self,
        data: pd.DataFrame,
        start_dt: str = None,
        end_dt: str = None,
        datafeed_cls: bt.feeds.PandasData = None,
    ) -> None:

        if start_dt is not None:
            data = data.loc[pd.to_datetime(start_dt):]

        if end_dt is not None:
            data = data.loc[:pd.to_datetime(end_dt)]

        if (start_dt is None) and (end_dt is None):
            start_dt, end_dt = data.index.min(), data.index.max()
            data = data.loc[start_dt:end_dt]

        self.datas = data

        logger.info("Loading data into backtest engine...")

        for code, df in track(
            data.groupby("code"),
            desc="Loading data into backtest engine..."
        ):
            df = df.drop(columns=["code"])
            df = check_dataframe_cols(df, datafeed_cls)
            assert "signal_1" in df.columns, "Missing signal_1 column"
            assert "signal_2" in df.columns, "Missing signal_2 column"
            assert "signal_3" in df.columns, "Missing signal_3 column"
            assert "vwap" in df.columns, "Missing vwap column"

            if df[["close", "signal_1", "signal_2","signal_3"]].dropna().empty:
                logger.warning(f"{code} close is all NaN, skipping...")
                continue

            datafeed = datafeed_cls(
                dataname=df.sort_index()
            )
            self.cerebro.adddata(datafeed, name=code)

        logger.success("Data loading completed!")

    def add_strategy(self, strategy: bt.Strategy, *args, **kwargs) -> None:
        self.cerebro.addstrategy(strategy, *args, **kwargs)