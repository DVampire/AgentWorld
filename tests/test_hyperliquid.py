"""Test script for Hyperliquid REST API client."""
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv
import asyncio
import json

import argparse
from mmengine import DictAction


root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

load_dotenv()

from src.environments.hyperliquidentry.client import HyperliquidClient
from src.environments.hyperliquidentry.websocket import HyperliquidWebSocket
from src.environments.hyperliquidentry.service import HyperliquidService
from src.environments.hyperliquidentry.types import GetDataRequest
from src.logger import logger
from src.config import config
from src.utils import get_env

def parse_args():
        parser = argparse.ArgumentParser(description='Online Trading Agent Example')
        parser.add_argument("--config", default=os.path.join(root, "configs", "online_trading_agent.py"), help="config file path")
        
        parser.add_argument(
            '--cfg-options',
            nargs='+',
            action=DictAction,
            help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file. If the value to '
            'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
            'Note that the quotation marks are necessary and that no white space '
            'is allowed.')
        args = parser.parse_args()
        return args


def test_get_exchange_info():
    """Test getting exchange information."""
    print("=" * 60)
    print("Testing Hyperliquid get_exchange_info()")
    print("=" * 60)
    
    # Initialize client (no private key needed for info endpoints)
    client = HyperliquidClient(
        wallet_address=os.getenv("HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS", ""),
        testnet=True  # Use testnet for testing
    )
    
    try:
        exchange_info = client.get_exchange_info()
        print(f"✅ Exchange info retrieved successfully")
        print(f"Keys: {list(exchange_info.keys())}")
        
        # Check universe
        universe = exchange_info.get('universe', [])
        print(f"✅ Universe contains {len(universe)} assets")
        
        # Show first few assets
        for idx, coin_info in enumerate(universe[:5]):
            if isinstance(coin_info, dict):
                coin_name = coin_info.get('name', '')
                print(f"  [{idx}] {coin_name}")
            else:
                print(f"  [{idx}] {coin_info}")
        
        return exchange_info
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_get_asset_index():
    """Test getting asset index from symbol."""
    print("\n" + "=" * 60)
    print("Testing Hyperliquid _get_asset_index()")
    print("=" * 60)
    
    client = HyperliquidClient(
        wallet_address=os.getenv("HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS", ""),
        testnet=True
    )
    
    try:
        # Test with BTC
        btc_index = client._get_asset_index("BTC")
        print(f"✅ BTC asset index: {btc_index}")
        
        # Test with ETH
        eth_index = client._get_asset_index("ETH")
        print(f"✅ ETH asset index: {eth_index}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_account():
    """Test getting account information."""
    print("\n" + "=" * 60)
    print("Testing Hyperliquid get_account()")
    print("=" * 60)
    
    wallet_address = os.getenv("HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS", "")
    if not wallet_address:
        print("⚠️  HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS not set, skipping account test")
        return None
    
    client = HyperliquidClient(
        wallet_address=wallet_address,
        testnet=True
    )
    
    try:
        account = client.get_account()
        print(f"✅ Account info retrieved successfully")
        print(f"Keys: {list(account.keys())}")
        print(f"Account: {account}")
        
        return account
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_get_positions():
    """Test getting positions."""
    print("\n" + "=" * 60)
    print("Testing Hyperliquid get_positions()")
    print("=" * 60)
    
    wallet_address = os.getenv("HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS", "")
    if not wallet_address:
        print("⚠️  HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS not set, skipping positions test")
        return None
    
    client = HyperliquidClient(
        wallet_address=wallet_address,
        testnet=True
    )
    
    try:
        positions = client.get_positions()
        print(f"✅ Positions retrieved successfully")
        print(f"Number of positions: {len(positions)}")
        
        for pos in positions[:5]:  # Show first 5 positions
            if isinstance(pos, dict):
                print(f"  Position: {pos}")
        
        return positions
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_create_order():
    """Test creating an order (market order)."""
    print("\n" + "=" * 60)
    print("Testing Hyperliquid create_order() - Market Order")
    print("=" * 60)
    
    wallet_address = os.getenv("HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS", "")
    private_key = os.getenv("HYPERLIQUID_TESTNET_TRADING_PRIVATE_KEY", "")
    
    if not wallet_address or not private_key:
        print("⚠️  HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS or HYPERLIQUID_TESTNET_TRADING_PRIVATE_KEY not set, skipping order test")
        return None
    
    client = HyperliquidClient(
        wallet_address=wallet_address,
        private_key=private_key,
        testnet=False  # Use testnet for testing
    )
    
    try:
        # Test market order
        print("Creating market order: BUY 0.01 BTC")
        result = client.create_order(
            symbol="BTC",
            side="buy",  # Will be converted to boolean
            order_type="Market",
            size=0.0001,
            stop_loss_price=100000.0,
            take_profit_price=110000.0
        )
        
        print(f"✅ Order created successfully")
        print(f"Result: {result}")
        
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_create_limit_order():
    """Test creating a limit order."""
    print("\n" + "=" * 60)
    print("Testing Hyperliquid create_order() - Limit Order")
    print("=" * 60)
    
    wallet_address = os.getenv("HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS", "")
    private_key = os.getenv("HYPERLIQUID_TESTNET_TRADING_PRIVATE_KEY", "")
    
    if not wallet_address or not private_key:
        print("⚠️  HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS or HYPERLIQUID_TESTNET_TRADING_PRIVATE_KEY not set, skipping limit order test")
        return None
    
    client = HyperliquidClient(
        wallet_address=wallet_address,
        private_key=private_key,
        testnet=True
    )
    
    try:
        # Get current price (approximate)
        # For testing, use a price that's likely to not fill immediately
        test_price = 50000.0  # Example price for BTC
        
        print(f"Creating limit order: BUY 0.01 BTC at {test_price}")
        result = client.create_order(
            symbol="BTC",
            side="buy",
            order_type="Limit",
            size=0.01,
            price=test_price,
            time_in_force="Gtc",
            reduce_only=False
        )
        
        print(f"✅ Limit order created successfully")
        print(f"Result: {result}")
        
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_websocket_candles(use_testnet: bool = False):
    """Test WebSocket candle data streaming (async version).
    
    Args:
        use_testnet: If True, use testnet. If False, use mainnet (more active).
    """
    network = "Testnet" if use_testnet else "Mainnet"
    print("\n" + "=" * 60)
    print(f"Testing Hyperliquid WebSocket Candle Stream ({network})")
    print("=" * 60)
    
    try:
        
        async def on_message(ws, channel: str, data: Dict):
            logger.info(f"📡 Received message: {channel} - {data}")
            
        async def on_error(ws, error: str):
            logger.error(f"❌ WebSocket error: {error}")
            
        async def on_close(ws):
            logger.info("🛑 WebSocket closed")
            
        async def on_open(ws):
            logger.info(f"✅ WebSocket opened ({network})")
        
        # Create WebSocket client without custom callbacks
        # The websocket.py already has built-in message handling
        ws_client = HyperliquidWebSocket(
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open,
            testnet=use_testnet
        )
        
        print("📋 WebSocket client created")
        
        # Subscribe to BTC and ETH candles
        print("📡 Subscribing to BTC and ETH 1-minute candles...")
        ws_client.subscribe_candle("BTC", interval="1m")
        ws_client.subscribe_candle("ETH", interval="1m")
        
        # Start WebSocket (in current async event loop)
        print("🚀 Starting WebSocket connection...")
        loop = asyncio.get_running_loop()
        ws_client.start(loop=loop)
        
        # Wait for data
        wait_seconds = 90  # Wait 90 seconds (at least 1 full minute)
        print(f"⏳ Waiting {wait_seconds} seconds for candle data...")
        print("   Note: Hyperliquid only sends candle data when a minute closes (every 60 seconds)")
        print(f"   {network} should have {'more' if not use_testnet else 'less'} activity")
        
        intervals = wait_seconds // 10
        for i in range(intervals):
            await asyncio.sleep(10)
            print(f"   ... {(i+1)*10}s elapsed (waiting for complete minute candle...)")
            
            # Check if task is still running
            if ws_client._task and ws_client._task.done():
                print(f"   ⚠️ WARNING: WebSocket task has stopped!")
                try:
                    exception = ws_client._task.exception()
                    print(f"   Task exception: {exception}")
                except:
                    pass
        
        # Stop WebSocket
        print("\n🛑 Stopping WebSocket...")
        await ws_client.stop()
        
        print(f"\n✅ Test completed. Check the logs above to see if candle data was received.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def test_complete_data_flow(use_testnet: bool = False):
    """Test complete data flow from WebSocket to database.
    
    Args:
        use_testnet: If True, use testnet. If False, use mainnet (more active).
    """
    network = "Testnet" if use_testnet else "Mainnet"
    print("\n" + "=" * 80)
    print(f"Testing Complete Hyperliquid Data Flow ({network})")
    print("=" * 80)
    
    # Configuration
    base_dir = Path(root) / "workdir" / "test_data_flow"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Test account (read-only, no private key needed for data stream)
    accounts = get_env("HYPERLIQUID_ACCOUNTS").get_secret_value()
    if accounts:
        accounts = json.loads(accounts)
    else:
        raise ValueError("HYPERLIQUID_ACCOUNTS not found")
    
    # Symbols to test
    test_symbols = ["BTC", "ETH"]
    
    print(f"\n📋 Configuration:")
    print(f"   Base Directory: {base_dir}")
    print(f"   Symbols: {test_symbols}")
    print(f"   Network: {network}")
    
    try:
        # Initialize service
        print("\n🚀 Step 1: Initializing Hyperliquid Service...")
        service = HyperliquidService(
            base_dir=str(base_dir),
            accounts=accounts,
            live=not use_testnet,  # Use mainnet for more active trading
            auto_start_data_stream=False,  # We'll start it manually
            symbol=test_symbols,
            data_type="candle"
        )
        
        await service.initialize()
        print("✅ Service initialized successfully")
        
        # Start data stream
        print("\n📡 Step 2: Starting data stream...")
        service.start_data_stream(test_symbols)
        print("✅ Data stream started")
        
        # Wait for data collection (need at least 60+ seconds for complete candle)
        wait_time = 90
        print(f"\n⏳ Step 3: Waiting {wait_time} seconds for data collection...")
        print("   (Hyperliquid pushes complete 1-minute candles every 60 seconds)")
        
        for i in range(wait_time // 10):
            await asyncio.sleep(10)
            print(f"   ... {(i+1)*10}s elapsed")
        
        # Query data from database
        print("\n🔍 Step 4: Querying data from database...")
        for symbol in test_symbols:
            request = GetDataRequest(
                symbol=symbol,
                data_type="candle",
                limit=5  # Get last 5 candles
            )
            
            result = await service.get_data(request)
            
            if result.success:
                data = result.extra.get("data", {})
                if symbol in data and "candle" in data[symbol]:
                    candles = data[symbol]["candle"]
                    print(f"\n✅ {symbol}: Found {len(candles)} candle(s)")
                    
                    # Show latest candle
                    if len(candles) > 0:
                        latest = candles[-1]
                        print(f"   Latest candle:")
                        print(f"      Timestamp: {latest.get('timestamp')}")
                        print(f"      Open: {latest.get('open')}")
                        print(f"      High: {latest.get('high')}")
                        print(f"      Low: {latest.get('low')}")
                        print(f"      Close: {latest.get('close')}")
                        print(f"      Volume: {latest.get('volume')}")
                else:
                    print(f"\n⚠️  {symbol}: No data found in database yet")
            else:
                print(f"\n❌ {symbol}: Failed to query data - {result.message}")
        
        # Stop data stream
        print("\n🛑 Step 5: Stopping data stream...")
        service.stop_data_stream()
        print("✅ Data stream stopped")
        
        # Cleanup
        print("\n🧹 Step 6: Cleaning up...")
        await service.cleanup()
        print("✅ Cleanup completed")
        
        print("\n" + "=" * 80)
        print("✅ Data Flow Test Completed Successfully!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during data flow test: {e}")
        import traceback
        traceback.print_exc()
        return None


async def async_main():
    """Run async tests."""
    print("=" * 60)
    print("Hyperliquid Client Test Suite (Async)")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test 1: Get exchange info
    exchange_info = test_get_exchange_info()
    
    # Test 2: Get asset index
    if exchange_info:
        test_get_asset_index()
    
    # Test 3: Get account
    test_get_account()
    
    # Test 4: Get positions
    test_get_positions()
    
    # Test 5: Create market order (commented out by default - requires private key)
    # Uncomment to test order creation
    test_create_order()
    
    # Test 6: Create limit order (commented out by default - requires private key)
    # Uncomment to test limit order creation
    # test_create_limit_order()
    
    # Test 7: WebSocket candle data stream
    # print("\n" + "=" * 60)
    # print("Starting WebSocket Test...")
    # print("=" * 60)
    # Use mainnet (testnet=False) because it has more active trading
    # await test_websocket_candles(use_testnet=False)
    
    # Test 8: Complete data flow (WebSocket -> Producer -> Database -> Consumer)
    # print("\n" + "=" * 60)
    # print("Starting Complete Data Flow Test...")
    # print("=" * 60)
    # # Use mainnet (testnet=False) because it has more active trading
    # await test_complete_data_flow(use_testnet=False)
    
    print("\n" + "=" * 60)
    print("Test Suite Completed")
    print("=" * 60)


async def main():
    
    args = parse_args()
    
    # Initialize configuration
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    await async_main()
    
    logger.info("| 🚪 Test completed")

if __name__ == "__main__":
    asyncio.run(main())
