import backtrader as bt
from .datafeed import CryptoDataFeed
from .engine import BackTesting
import pandas as pd
from typing import Dict
from typing import Any, Dict, List, Union
from .base_types import BaseStrategy,BaseSignal
from .utils import get_strategy_cumulative_return,get_strategy_maxdrawdown,get_strategy_sharpe_ratio,plot_cumulative_return
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

__all__ = ["run_strategy", "COMMISSION"]
# 设置初始金额及手续费
COMMISSION: Dict = dict(
    cash=1e8, commission=5e-4,slippage_perc=0.0,leverage=1.0
)

# 设置策略参数
STRATEGY_PARAMS: Dict = {"verbose": False, "hold_num": 1, "leverage": 1.0}



class TotalCommission(bt.Analyzer):
    def __init__(self):
        self.total_commission = 0.0
        self.by_data = defaultdict(float)
        self.n_trades = 0

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.n_trades += 1
        self.total_commission += trade.commission
        self.by_data[trade.data._name] += trade.commission

    def get_analysis(self):
        return {
            "total_commission": self.total_commission,
            "by_data": dict(self.by_data),
            "n_closed_trades": self.n_trades,
        }

def update_params(default_params: Dict, custom_params: Dict) -> Dict:
    if custom_params is None:
        return default_params
    default_params.update(custom_params)
    return default_params


def backtest_strategy(
    data: pd.DataFrame,
    code: str,
    strategy: bt.Strategy,
    strategy_kwargs: Dict = {},
    commission_kwargs: Dict = {},
    number_of_signals: int = 3,
):
    commission_kwargs: Dict = update_params(COMMISSION, commission_kwargs)
    strategy_kwargs: Dict = update_params(STRATEGY_PARAMS, strategy_kwargs)
    assert commission_kwargs["leverage"] == strategy_kwargs["leverage"], "leverage must be same in both strategy and commission settings"

    if isinstance(code, str):

        df: pd.DataFrame = data.query("code == @code").copy()

    elif isinstance(code, list):

        df: pd.DataFrame = data.query("code in @code").copy()
        strategy_kwargs["hold_num"] = len(code)

    
    signals = [f"signal_{i+1}" for i in range(number_of_signals)]
    required_columns = ["close","vwap"] + signals

    df: pd.DataFrame = df.dropna(subset=required_columns)
    bt_engine = BackTesting(**commission_kwargs)
    bt_engine.load_data(
        df,
        datafeed_cls=CryptoDataFeed,
    )
    print(strategy_kwargs)
    bt_engine.add_strategy(strategy, **strategy_kwargs)
    bt_engine.cerebro.addanalyzer(
        bt.analyzers.TimeReturn, _name="time_return", timeframe=bt.TimeFrame.Days
    )
    bt_engine.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    bt_engine.cerebro.addanalyzer(TotalCommission, _name="total_commission")
    tmp_result = bt_engine.cerebro.run()


    try:
        if not tmp_result:
            raise RuntimeError("Zero trade were made")
        result = tmp_result[0]
    except Exception as e:
        raise RuntimeError(f"Zero trade were made")
    
    return result




def test_backtest(
        data: pd.DataFrame,
        code: Union[str, List[str]],
        strategy: BaseStrategy,
        signal: BaseSignal,
        strategy_kwargs: Dict = STRATEGY_PARAMS,
        commission_kwargs: Dict = COMMISSION,
        workdir: Path = None,
    ) -> Any:
        """Run backtest"""
        combo_data: pd.DataFrame = signal(data).fit(data)
        combo_data.set_index("trade_time", inplace=True)

        result = backtest_strategy(
            data=combo_data,
            code=code,
            strategy=strategy,
            strategy_kwargs=strategy_kwargs,
            commission_kwargs=commission_kwargs,
        )
        ax = plot_cumulative_return(result, title="Buy and Hold Strategy")
        plt.savefig(workdir / "cumulative_return.png") if workdir else None
        plt.close(ax.figure)
        return {
            "sharpe_ratio": get_strategy_sharpe_ratio(result),
            "cumulative_return (%)": get_strategy_cumulative_return(result).iloc[-1],
            "max_drawdown (%)": get_strategy_maxdrawdown(result),
            "picture_path": str(workdir / "cumulative_return.png") if workdir else None
        }