from pathlib import Path
import signal
import sys
from tracemalloc import start
from turtle import delay
from typing import Any, Dict, List
from dotenv import load_dotenv
from duckdb import df
from matplotlib import pyplot as plt
load_dotenv(verbose=True)
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)
from src.environment.quickbacktest.utils import (
    get_excess_return, 
    get_strategy_cumulative_return, 
    get_strategy_maxdrawdown, 
    get_strategy_sharpe_ratio, 
    get_strategy_total_commission, 
    plot_cumulative_return,
    get_strategy_win_rate,
    get_relative_equity_curve,
    path_outperformance_score
)
from src.environment.quickbacktest.backtest import backtest_strategy
from libs.BinanceDatabase.src.core import BinanceDatabase
from libs.BinanceDatabase.src.core.time_utils import utc_ms
from src.environment.quickbacktest.base_types import BaseStrategyEvaluation,BaseSignal,SignalData,SignalRegistry,SignalPipeline
from datetime import datetime
import pandas as pd
import importlib 
from pathlib import Path
import importlib.util
import sys
from typing import Type
from loguru import logger as trade_logger
from functools import lru_cache

def dict_to_markdown_table(d: dict) -> str:
    headers = "| Key | SubKey | Value |\n|-----|--------|-------|\n"
    rows = ""

    for k, v in d.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                rows += f"| {k} | {sub_k} | {sub_v} |\n"
        else:
            rows += f"| {k} |  | {v} |\n"

    return headers + rows

class ClassLoader:
    @staticmethod
    def load_class(file_path: str | Path, class_name: str) -> Type:
        file_path = Path(file_path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        # ⚠️ module_name 必须唯一，防止 sys.modules 冲突
        module_name = f"_dynamic_{file_path.stem}_{hash(file_path)}"

        spec = importlib.util.spec_from_file_location(
            module_name,
            str(file_path),
        )
        if spec is None or spec.loader is None:
            raise ImportError(file_path)

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, class_name):
            raise AttributeError(
                f"Class '{class_name}' not found in {file_path}"
            )

        return getattr(module, class_name)


STRATEGY_PARAMS_ENV: Dict = {"verbose": False, "hold_num": 1, "leverage": 1.0}
COMMISSION_ENV: Dict = dict(
    cash=1e8, commission=0.0004,slippage_perc=0.0001,leverage=1.0
)

INTERVAL = "1d"

# def get_file_hash(path):
#     import hashlib
#     with open(path, "rb") as f:
#         return hashlib.md5(f.read()).hexdigest()
    

def signal_to_dataframe(data_dir,watermark_dir,venue,symbol,start_ms,end_ms,slot_modules: List[str], base_dir) -> pd.DataFrame:

    svc = BinanceDatabase(data_root=data_dir,state_db=watermark_dir)

    data = svc.query(venue=venue, symbol=symbol, start_ms=start_ms, end_ms=end_ms,as_="pandas",columns=["open_time","symbol","open","high","low","close","volume","quote_volume"],interval=INTERVAL)
    data["trade_time"] = pd.to_datetime(data["open_time"], unit='ms', utc=True)
    data.rename(columns={"symbol":"code","quote_volume":"amount"}, inplace=True)
    data.drop(columns=["open_time"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    signal_data = SignalData(data)
    registry = SignalRegistry()
    slot_modules = {f"signal_{i+1}": module_name for i, module_name in enumerate(slot_modules)}

    for slot_name, signal_module in slot_modules.items():
        SignalCls = ClassLoader.load_class(
            file_path=Path(base_dir) / "signals" / f"{signal_module}.py",
            class_name=signal_module,
        )
        registry.assign(slot_name, SignalCls(signal_data))

    pipeline = SignalPipeline(registry)
    combo_data = pipeline.run(
        include_vwap=True,
    )

    del signal_data
    del registry
    del pipeline
    del SignalCls

    combo_data.set_index("trade_time", inplace=True)
    return combo_data



def run_backtest(data_dir: str = None, watermark_dir: str = None, venue: str = None, symbol: str = None,start: datetime = None,end: datetime = None,strategy_module: str = "strategy_template", signal_modules: List[str] = ["signal_template"],base_dir: str = None,slippage_perc: float = None,plot=True) -> Any:
    
    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023,1,1))

    AgentStrategy = ClassLoader.load_class(
        file_path=Path(base_dir) / "strategies" / f"{strategy_module}.py",
        class_name=strategy_module,
    )

    COMMISSION_ENV["slippage_perc"] = slippage_perc if slippage_perc is not None else COMMISSION_ENV["slippage_perc"]
    combo_data = signal_to_dataframe(data_dir,watermark_dir,venue,symbol,start_ms,end_ms,signal_modules, base_dir)
    result = backtest_strategy(
        data=combo_data,
        code=symbol,
        strategy=AgentStrategy,
        strategy_kwargs=STRATEGY_PARAMS_ENV,
        commission_kwargs=COMMISSION_ENV,
    )
    
    if plot:
        ax = plot_cumulative_return(result,combo_data.query("code==@symbol")["close"], title=strategy_module + ' '+ str(signal_modules))
        save_path = Path(base_dir) /"images" / f"{strategy_module}_{signal_modules}_{start.date()}_{end.date()}_cumulative_return.png"
        plt.savefig(save_path)
        plt.close(ax.figure)
    return {
        "sharpe_ratio": get_strategy_sharpe_ratio(result),
        "cumulative_return (%)": get_strategy_cumulative_return(result).iloc[-1]*100,
        "max_drawdown (%)": get_strategy_maxdrawdown(result)*100,
        "win_rate (%)": get_strategy_win_rate(result).iloc[0]['win_rate']*100,
        "closed_trades": get_strategy_win_rate(result).iloc[0]['closed'],
        "total_commission (%)": get_strategy_total_commission(result)/COMMISSION_ENV["cash"] * 100,
        "excess_return_ratio (%)": get_excess_return(
            result,
            combo_data.query("code==@symbol")["close"],
            benchmark_is_return=False,
        )*100,
        # "max_shortfall (%)": -path_outperformance_score(get_relative_equity_curve(result,combo_data.query("code==@symbol")["close"])["W_rel"],mode="max_shortfall")*100,
        # "cumulative_return_path": str(save_path) if base_dir else None
    }

# def _hit_rate(signal: pd.Series, close: pd.Series, ma_window: int = 20) -> float:
#     """Calculate hit rate using MA(close, 20) returns.

#     Hit rate is defined as the proportion of times the signal correctly predicts
#     the direction of the next period's return, where return is computed on the
#     moving-averaged close series.

#     Args:
#         signal: Series containing the signal values.
#         close: Close price series.
#         ma_window: Moving average window for close (default=20).

#     Returns:
#         Hit rate as a float between 0 and 1.
#     """
#     close_ma = close.rolling(ma_window).mean()
#     returns = close_ma.pct_change().shift(-1)

#     correct = ((signal > 0) & (returns > 0)) | ((signal < 0) & (returns < 0))

#     # 避免 MA 前 ma_window-1 个 NaN 影响：只在 returns 非 NaN 的地方计数
#     valid = returns.notna() & signal.notna() & (signal != 0)
#     if valid.sum() == 0:
#         return float("nan")

#     hit_rate = correct[valid].mean()  # True/False 的 mean 就是命中率
#     return float(hit_rate)

    


def get_signal_quantile(data_dir: str = None, watermark_dir: str = None, venue: str = None, symbol: str = None,start: datetime = None,end: datetime = None, signal_module: str = "signal_template",base_dir: str = None) -> Any:

    svc = BinanceDatabase(data_root=data_dir,state_db=watermark_dir)
    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023,1,1))

    data = svc.query(venue=venue, symbol=symbol, start_ms=start_ms, end_ms=end_ms,as_="pandas",columns=["open_time","symbol","open","high","low","close","volume","quote_volume"],interval=INTERVAL)
    data["trade_time"] = pd.to_datetime(data["open_time"], unit='ms', utc=True)
    data.rename(columns={"symbol":"code","quote_volume":"amount"}, inplace=True)
    data.drop(columns=["open_time"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    SignalCls = ClassLoader.load_class(
            file_path=Path(base_dir) / "signals" / f"{signal_module}.py",
            class_name=signal_module,
    )

    signal_data = SignalData(data)
    signal_result = SignalCls(signal_data).compute()
    del signal_data

    result  = {}

    # for factor in signals:
    #     hit_rate = _hit_rate(combo_data[factor], combo_data["close"])
    #     factors_value[factor]["hit_rate"] = hit_rate


    result["range"] = signal_result.describe().to_dict()[symbol]

    return result

def run_signal_evaluation(data_dir: str = None, watermark_dir: str = None, venue: str = None, symbol: str = None,start: datetime = None,end: datetime = None, signal_module: List[str] = None ,evaluation_module: str = "evaluation_template",base_dir: str = None) -> Any:

    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023,1,1))

    combo_data = signal_to_dataframe(data_dir,watermark_dir,venue,symbol,start_ms,end_ms,signal_module, base_dir)

    AgentBenchmark = ClassLoader.load_class(
        file_path=Path(base_dir) / "evaluations" / f"{evaluation_module}.py",
        class_name=evaluation_module,
    )


    benchmark_instance = AgentBenchmark(data = combo_data,base_dir=base_dir)
    numeric_result = benchmark_instance.numeric_evaluation()

    plot_result = benchmark_instance.plot_evaluation()

    return {"numeric_result": numeric_result, "plot_result": plot_result}


def run_strategy_evaluation(data_dir: str = None, watermark_dir: str = None, venue: str = None, symbol: str = None,start: datetime = None,end: datetime = None,signal_module:List[str] = None, strategy_module: str = "strategy_template", evaluation_module: str = "evaluation_template",base_dir: str = None,slippage_perc: float = None) -> Any:

    run_backtest(data_dir, watermark_dir, venue, symbol, start, end, strategy_module, signal_module, base_dir, slippage_perc, plot=False)

    AgentStrategyBenchmark:BaseStrategyEvaluation = ClassLoader.load_class(
        file_path=Path(base_dir) / "evaluations" / f"{evaluation_module}.py",
        class_name=evaluation_module,
    )

    benchmark_instance = AgentStrategyBenchmark(base_dir=base_dir)
    return benchmark_instance.run()


def get_rank_ic(data_dir: str = None, watermark_dir: str = None, venue: str = None, symbol: str = None,start: datetime = None,end: datetime = None, signal_module: str = "signal_template",base_dir: str = None,horizon: int = 1) -> Any:

    svc = BinanceDatabase(data_root=data_dir,state_db=watermark_dir)
    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023,1,1))

    data = svc.query(venue=venue, symbol=symbol, start_ms=start_ms, end_ms=end_ms,as_="pandas",columns=["open_time","symbol","open","high","low","close","volume","quote_volume"],interval=INTERVAL)
    data["trade_time"] = pd.to_datetime(data["open_time"], unit='ms', utc=True)
    data.rename(columns={"symbol":"code","quote_volume":"amount"}, inplace=True)
    data.drop(columns=["open_time"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    SignalCls = ClassLoader.load_class(
            file_path=Path(base_dir) / "signals" / f"{signal_module}.py",
            class_name=signal_module,
    )

    signal_data = SignalData(data)
    signal_result = SignalCls(signal_data).compute()[symbol]

    close = signal_data.close[symbol]

    del signal_data

    df = pd.concat([signal_result.rename("signal"), close.rename("close")], axis=1).dropna()


    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1

    df = df.dropna()

    if len(df) < 2:
        return float("nan")

    return df["signal"].corr(df["future_return"], method="spearman")



def get_rank_ic(
    data_dir: str = None,
    watermark_dir: str = None,
    venue: str = None,
    symbol: str = None,
    start: datetime = None,
    end: datetime = None,
    signal_module: str = "signal_template",
    base_dir: str = None,
    horizon: int = 1,
    ic_window: int = 60,
) -> Any:
    svc = BinanceDatabase(data_root=data_dir, state_db=watermark_dir)
    start_ms = utc_ms(start) if start else utc_ms(datetime(2022, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2023, 1, 1))

    data = svc.query(
        venue=venue,
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        as_="pandas",
        columns=["open_time", "symbol", "open", "high", "low", "close", "volume", "quote_volume"],
        interval=INTERVAL,
    )
    data["trade_time"] = pd.to_datetime(data["open_time"], unit="ms", utc=True)
    data.rename(columns={"symbol": "code", "quote_volume": "amount"}, inplace=True)
    data.drop(columns=["open_time"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    SignalCls = ClassLoader.load_class(
        file_path=Path(base_dir) / "signals" / f"{signal_module}.py",
        class_name=signal_module,
    )

    signal_data = SignalData(data)
    signal_result = SignalCls(signal_data).compute()[symbol]
    close = signal_data.close[symbol]

    del signal_data

    df = pd.concat([signal_result.rename("signal"), close.rename("close")], axis=1).dropna()
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df = df.dropna()

    if len(df) < ic_window:
        return {
            "rank_ic": float("nan"),
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "icir": float("nan"),
        }

    rank_ic = df["signal"].corr(df["future_return"], method="spearman")

    rolling_ic = []
    for i in range(ic_window, len(df) + 1):
        sub = df.iloc[i - ic_window : i]
        ic = sub["signal"].corr(sub["future_return"], method="spearman")
        rolling_ic.append(ic)

    ic_series = pd.Series(rolling_ic).dropna()

    if len(ic_series) < 2:
        ic_mean = float("nan")
        ic_std = float("nan")
        icir = float("nan")
    else:
        ic_mean = ic_series.mean()
        ic_std = ic_series.std(ddof=1)
        icir = float("nan") if ic_std == 0 else ic_mean / ic_std

    return {
        "rank_ic": rank_ic,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
    }