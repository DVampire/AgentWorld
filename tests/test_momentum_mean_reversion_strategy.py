"""Test script for Momentum and Mean Reversion Hybrid Strategy with real Hyperliquid API data."""

import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import pandas as pd

root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

load_dotenv()

# Direct import to avoid triggering src.tools global initialization
from src.tools.factor_strategy.momentum_mean_reversion_strategy import MomentumMeanReversionStrategy
from src.environments.hyperliquidentry.client import HyperliquidClient

# Use standard logging instead of src.logger to avoid triggering other imports
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def get_hyperliquid_candles(symbol: str = "BTC", limit: int = 100) -> pd.DataFrame:
    """Get candle data directly from Hyperliquid API (real-time only)."""
    client = HyperliquidClient(wallet_address="", private_key=None, testnet=False)
    
    now_time = int(time.time() * 1000)
    start_time = int(now_time - (limit + 10) * 60 * 1000)
    end_time = int(now_time)
    
    logger.info(f"| 📡 Fetching {limit} candles from Hyperliquid API...")
    symbol_data = await client.get_symbol_data(symbol, start_time=start_time, end_time=end_time)
    
    if not symbol_data:
        raise Exception(f"API returned no data for {symbol}")
    
    logger.info(f"| ✅ API returned {len(symbol_data)} candles")
    
    # Convert to DataFrame
    candles = [{
        "timestamp": pd.Timestamp(c.get("T", c.get("t", 0)), unit="ms"),
        "open": float(c.get("o", 0)),
        "high": float(c.get("h", 0)),
        "low": float(c.get("l", 0)),
        "close": float(c.get("c", 0)),
        "volume": float(c.get("v", 0)),
    } for c in symbol_data]
    
    df = pd.DataFrame(candles).sort_values("timestamp")
    if len(df) > limit:
        df = df.tail(limit)
    
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    logger.info(f"| ✅ Prepared {len(df)} candles")
    logger.info(f"|    Price: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    return df


async def main():
    """Test Momentum and Mean Reversion Hybrid Strategy with real Hyperliquid data."""
    print("=" * 80)
    print("Testing Momentum and Mean Reversion Hybrid Strategy with Hyperliquid Data")
    print("=" * 80)
    
    try:
        # Get data
        print("\n" + "-" * 80)
        print("Step 1: Fetching data from Hyperliquid API")
        print("-" * 80)
        
        df = await get_hyperliquid_candles(symbol="BTC", limit=100)
        print(f"✅ Retrieved {len(df)} candles")
        print(f"   Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
        
        # Initialize strategy
        print("\n" + "-" * 80)
        print("Step 2: Running Strategy Analysis")
        print("-" * 80)
        
        strategy = MomentumMeanReversionStrategy()
        market_description = strategy.analyze(df)
        
        # Display results
        print("\n" + "-" * 80)
        print("Step 3: Market Analysis (Natural Language Description)")
        print("-" * 80)
        print(f"\n{market_description}")
        
        print("\n" + "=" * 80)
        print("✅ Test completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        # Force exit to handle WebSocket connections that don't close properly
        import os
        os._exit(0)
