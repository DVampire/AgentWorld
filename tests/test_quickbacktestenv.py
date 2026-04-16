from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environment import QuickBacktestEnvironment
import asyncio

env = QuickBacktestEnvironment(base_dir="workdir/trading_strategy_agent/environment/quick_backtest")

async def setup_environment():
    await env.initialize()

    signal_list = await env.listModules(module_type="signals")
    result = await env.backtest(strategy_name="VolAdaptiveMR_v12", signal_name="AgentSignal", rolling_window=1440)

    return signal_list,result

print(asyncio.run(setup_environment()))