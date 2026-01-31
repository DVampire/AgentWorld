from pathlib import Path
import sys
from typing import Any, Dict
from dotenv import load_dotenv
from matplotlib import pyplot as plt
load_dotenv(verbose=True)
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)
from src.environment.quickbacktest.utils import get_excess_return, get_strategy_cumulative_return, get_strategy_maxdrawdown, get_strategy_sharpe_ratio, get_strategy_total_commission, plot_cumulative_return,get_strategy_win_rate
from src.environment.quickbacktest.backtest import backtest_strategy
from libs.BinanceDatabase.src.core import BinanceDatabase
from libs.BinanceDatabase.src.core.time_utils import utc_ms
from datetime import datetime
import pandas as pd
import importlib 
from pathlib import Path
import importlib.util
import sys
from typing import Type


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
    cash=1e8, commission=0.00015,slippage_perc=0.0001,leverage=1.0
)

def run_backtest(data_dir: str = None, watermark_dir: str = None, venue: str = None, symbol: str = None,start: datetime = None,end: datetime = None,strategy_module: str = "strategy_template", signal_module: str = "signal_template",base_dir: str = None) -> Any:
    svc = BinanceDatabase(data_root=data_dir,state_db=watermark_dir)

    start_ms = utc_ms(start) if start else utc_ms(datetime(2024, 1, 1))
    end_ms = utc_ms(end) if end else utc_ms(datetime(2025, 1, 1))

    AgentStrategy = ClassLoader.load_class(
        file_path=Path(base_dir) / "strategies" / f"{strategy_module}.py",
        class_name=strategy_module,
    )
    AgentSignal = ClassLoader.load_class(
        file_path=Path(base_dir) / "signals" / f"{signal_module}.py",
        class_name=signal_module,
    )
    data = svc.query(venue=venue, symbol=symbol, start_ms=start_ms, end_ms=end_ms,as_="pandas",columns=["open_time","symbol","open","high","low","close","volume","quote_volume"],interval="1m")
    data["trade_time"] = pd.to_datetime(data["open_time"], unit='ms', utc=True)
    data.rename(columns={"symbol":"code","quote_volume":"amount"}, inplace=True)
    data.drop(columns=["open_time"], inplace=True)
    data.reset_index(drop=True, inplace=True)

    combo_data: pd.DataFrame = AgentSignal(data).fit()
    combo_data.set_index("trade_time", inplace=True)
    result = backtest_strategy(
        data=combo_data,
        code=symbol,
        strategy=AgentStrategy,
        strategy_kwargs=STRATEGY_PARAMS_ENV,
        commission_kwargs=COMMISSION_ENV,
    )
    

    ax = plot_cumulative_return(result,combo_data.query("code==@symbol")["close"], title=strategy_module + ' '+ signal_module)
    save_path = Path(base_dir) / "cumulative_return.png"
    plt.savefig(save_path)
    plt.close(ax.figure)
    return {
        "sharpe_ratio": get_strategy_sharpe_ratio(result),
        "cumulative_return (%)": get_strategy_cumulative_return(result).iloc[-1]*100,
        "max_drawdown (%)": get_strategy_maxdrawdown(result)*100,
        "win_rate (%)": get_strategy_win_rate(result).iloc[0]['win_rate']*100,
        "total_commission (%)": get_strategy_total_commission(result)/COMMISSION_ENV["cash"] * 100,
        "excess_return_ratio (%)": get_excess_return(
            result,
            combo_data.query("code==@symbol")["close"],
            benchmark_is_return=False,
        )*100,
        # "cumulative_return_path": str(save_path) if base_dir else None
    }
