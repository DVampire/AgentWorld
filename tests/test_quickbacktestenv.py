from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environment import QuickBacktestEnvironment
import asyncio

env = QuickBacktestEnvironment(base_dir="workdir/trading_strategy_agent/environment/quickbacktest")

async def setup_environment():
    await env.initialize()

    result = await env.getSignalQuantile(signal_name="AgentSignal")
    signal_list = await env.listModules(module_type="signals")

    return signal_list,result

print(asyncio.run(setup_environment()))