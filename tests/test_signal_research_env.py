from pathlib import Path
import sys
from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.environment import SignalResearchEnvironment
import asyncio

env = SignalResearchEnvironment(base_dir="workdir/trading_strategy_agent/environment/signal_research")

async def setup_environment():
    await env.initialize()
    result = await env.listanalysisToolcases()



    quantile = await env.getSignalQuantile(
        module_name="VolTrendDiff",
        start="2023-04-12 00:00",
        end="2023-06-16 00:00"
    )
    return quantile

print(asyncio.run(setup_environment()))