"""Test script for Hyperliquid REST API client."""
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv
import asyncio
import time
import json

from hyperliquid.info import Info

import argparse
from mmengine import DictAction


root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

load_dotenv()

from src.environments.hyperliquidentry.client import HyperliquidClient
from src.environments.hyperliquidentry.service import HyperliquidService
from src.environments.hyperliquidentry.types import GetDataRequest, CreateOrderRequest, OrderType
from src.environments.hyperliquid_environment import HyperliquidEnvironment
from src.logger import logger
from src.config import config
from src.utils import get_env
from src.environments import ecp
from src.utils import get_standard_timestamp
from src.environments.database.service import DatabaseService
from src.environments.database.types import QueryRequest, SelectRequest

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


async def test_client():
    """Test HyperliquidClient basic functionality."""
    print("=" * 80)
    print("Testing HyperliquidClient")
    print("=" * 80)
    
    wallet_address = os.getenv("HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS", "")
    testnet = False
    
    # Initialize client (no private key needed for info endpoints)
    client = HyperliquidClient(
        wallet_address=wallet_address if wallet_address else "",
        testnet=testnet
    )
    
    results = {}
    
    # Test 1: Get exchange info
    print("\n" + "-" * 80)
    print("📋 Test 1: get_exchange_info()")
    print("-" * 80)
    try:
        exchange_info = await client.get_exchange_info()
        print(f"✅ Exchange info retrieved successfully")
        print(f"| 📝 Exchange info: \n{json.dumps(exchange_info, indent=4)}")
        
        # Check universe
        universe = exchange_info.get('universe', [])
        print(f"✅ Universe contains {len(universe)} assets")
        
        # Show first few assets
        for idx, coin_info in enumerate(universe[:5]):
            if isinstance(coin_info, dict):
                coin_name = coin_info.get('name', '')
                print(f"   [{idx}] {coin_name}")
            else:
                print(f"   [{idx}] {coin_info}")
        
        results["exchange_info"] = exchange_info
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results["exchange_info"] = None
    
    # Test 2: Get symbol info
    print("\n" + "-" * 80)
    print("🔍 Test 2: get_symbol_info()")
    print("-" * 80)
    try:
        # Test with BTC
        btc_symbol_info = await client.get_symbol_info("BTC")
        print(f"✅ BTC symbol info: \n{json.dumps(btc_symbol_info, indent=4)}")
        
        # Test with ETH
        eth_symbol_info = await client.get_symbol_info("ETH")
        print(f"✅ ETH symbol info: \n{json.dumps(eth_symbol_info, indent=4)}")
        
        results["symbol_info"] = {"BTC": btc_symbol_info, "ETH": eth_symbol_info}
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results["symbol_info"] = None
    
    # Test 3: Get account (requires wallet address)
    print("\n" + "-" * 80)
    print("👤 Test 3: get_account()")
    print("-" * 80)
    if not wallet_address:
        print("⚠️  HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS not set, skipping account test")
        results["account"] = None
    else:
        try:
            account = await client.get_account()
            print(f"✅ Account info retrieved successfully")
            print(f"| 📝 Account: \n{json.dumps(account, indent=4)}")
            results["account"] = account
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results["account"] = None
    
    # Test 4: Get positions (requires wallet address)
    print("\n" + "-" * 80)
    print("📊 Test 4: get_positions()")
    print("-" * 80)
    if not wallet_address:
        print("⚠️  HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS not set, skipping positions test")
        results["positions"] = None
    else:
        try:
            positions = await client.get_positions()
            print(f"✅ Positions retrieved successfully")
            print(f"| 📝 Positions: \n{json.dumps(positions, indent=4)}")
            results["positions"] = positions
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results["positions"] = None
    
    # Test 5: Get orders (requires wallet address)
    print("\n" + "-" * 80)
    print("📊 Test 5: get_orders()")
    print("-" * 80)
    if not wallet_address:
        print("⚠️  HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS not set, skipping orders test")
        results["orders"] = None
    else:
        try:
            orders = await client.get_orders()
            print(f"✅ Orders retrieved successfully")
            print(f"| 📝 Orders: \n{json.dumps(orders, indent=4)}")
            results["orders"] = orders
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results["orders"] = None
            
    
    # Test 6: Get symbol data
    print("\n" + "-" * 80)
    print("🔍 Test 6: get_symbol_data()")
    print("-" * 80)
    try:
        while True:
            # Wait until the next minute boundary to ensure fresh candle data
            now = time.time()
            seconds_until_next_minute = 60 - (now % 60)
            if seconds_until_next_minute > 0 and seconds_until_next_minute < 60:
                await asyncio.sleep(seconds_until_next_minute)
            
            now_time = int(time.time() * 1000)
            start_time = int(now_time - 120 * 60 * 1000) # 2 hours ago
            end_time = int(now_time)
            symbol_data = await client.get_symbol_data("BTC", start_time=start_time, end_time=end_time)
            
            print(f"✅ Symbol data retrieved successfully")
            for item in symbol_data:
                
                current_time = int(time.time()) * 1000
                open_time = item['t']
                close_time = item['T']
                
                current_time_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(current_time / 1000))
                current_time_local = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time / 1000))
                open_time_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(open_time / 1000))
                close_time_utc = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(close_time / 1000))
                open_time_local = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(open_time / 1000))
                close_time_local = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(close_time / 1000))
                
                print(f"   Current Timestamp: {current_time}")
                print(f"   Current Timestamp (UTC): {current_time_utc}")
                print(f"   Current Timestamp (Local): {current_time_local}")
                print(f"   Open Time (ms): {open_time}")
                print(f"   Close Time (ms): {close_time}")
                print(f"   Open Time (UTC): {open_time_utc}")
                print(f"   Close Time (UTC): {close_time_utc}")
                print(f"   Open Time (Local): {open_time_local}")
                print(f"   Close Time (Local): {close_time_local}")
                print(f"   Open: {item['o']}")
                print(f"   High: {item['h']}")
                print(f"   Low: {item['l']}")  
                print(f"   Close: {item['c']}")
                print(f"   Volume: {item['v']}")
                print()
            results["symbol_data"] = symbol_data
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results["symbol_data"] = None
    
    print("\n" + "=" * 80)
    print("✅ Client tests completed")
    print("=" * 80)
    
    return results

async def test_orders():
    """Test order creation at Client, Service, and Environment layers."""
    print("\n" + "=" * 80)
    print("Testing Order Creation at All Layers (Client -> Service -> Environment)")
    print("=" * 80)
    
    wallet_address = os.getenv("HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS", "")
    private_key = os.getenv("HYPERLIQUID_TESTNET_TRADING_PRIVATE_KEY", "")
    
    if not wallet_address or not private_key:
        print("⚠️  HYPERLIQUID_TESTNET_TRADING_WALLET_ADDRESS or HYPERLIQUID_TESTNET_TRADING_PRIVATE_KEY not set, skipping test")
        return None
    
    # Test parameters
    symbol = "BTC"
    qty = 1e-4  # Small quantity for testing
    stop_loss_price = 90000.0
    take_profit_price = 120000.0
    
    results = {
        "client": None,
        "service": None,
        "environment": None
    }
    
    try:
        # ========== Test 1: Client Layer ==========
        print("\n" + "-" * 80)
        print("📦 Test 1: Client Layer (HyperliquidClient.create_order)")
        print("-" * 80)
        
        client = HyperliquidClient(
            wallet_address=wallet_address,
            private_key=private_key,
            testnet=False
        )
        
        print(f"Creating order via Client: BUY {qty} {symbol} (Market)")
        client_result = await client.create_order(
            symbol=symbol,
            side="buy",
            order_type="Market",
            size=qty,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
        
        print(f"✅ Client order result:")
        print(f"   Keys: {list(client_result.keys())}")
        if 'main_order' in client_result:
            main_order = client_result.get('main_order', {})
            print(f"   Main order status: {main_order.get('status', 'N/A')}")
            if main_order.get('status') == 'ok':
                response = main_order.get('response', {})
                if response.get('type') == 'order':
                    data = response.get('data', {})
                    statuses = data.get('statuses', [])
                    if statuses:
                        status = statuses[0]
                        if 'filled' in status:
                            print(f"   Order ID: {status['filled'].get('oid', 'N/A')}")
                            print(f"   Status: filled")
                        elif 'resting' in status:
                            print(f"   Order ID: {status['resting'].get('oid', 'N/A')}")
                            print(f"   Status: resting (submitted)")
                        elif 'error' in status:
                            print(f"   Error: {status.get('error', 'Unknown')}")
        
        if 'stop_loss_order' in client_result and client_result['stop_loss_order']:
            print(f"   Stop loss order: Created")
        if 'take_profit_order' in client_result and client_result['take_profit_order']:
            print(f"   Take profit order: Created")
        
        results["client"] = client_result
        
        # Wait a bit between orders
        await asyncio.sleep(2)
        
        # ========== Test 2: Service Layer ==========
        print("\n" + "-" * 80)
        print("🔧 Test 2: Service Layer (HyperliquidService.create_order)")
        print("-" * 80)
        
        # Setup service
        base_dir = Path(root) / "workdir" / "test_order_layers"
        base_dir.mkdir(parents=True, exist_ok=True)
        
        accounts = [{
            "name": "TestAccount",
            "address": wallet_address,
            "private_key": private_key
        }]
        
        service = HyperliquidService(
            base_dir=str(base_dir),
            accounts=accounts,
            live=True,  # Use testnet
            auto_start_data_stream=False,
            symbol=[symbol],
            data_type=["candle"]
        )
        await service.initialize()
        
        print(f"Creating order via Service: BUY {qty} {symbol} (Market)")
        service_request = CreateOrderRequest(
            account_name="TestAccount",
            symbol=symbol,
            side="buy",
            order_type=OrderType.MARKET,
            qty=qty,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
        
        service_result = await service.create_order(service_request)
        
        print(f"✅ Service order result:")
        print(f"   Success: {service_result.success}")
        print(f"   Message: {service_result.message}")
        if service_result.extra:
            order_info = service_result.extra.get('order_info', {})
            print(f"   Order ID: {order_info.get('order_id', 'N/A')}")
            print(f"   Order Status: {order_info.get('order_status', 'N/A')}")
            if order_info.get('stop_loss_order'):
                print(f"   Stop Loss Order: Created")
            if order_info.get('take_profit_order'):
                print(f"   Take Profit Order: Created")
            if order_info.get('stop_loss_error'):
                print(f"   Stop Loss Error: {order_info.get('stop_loss_error')}")
            if order_info.get('take_profit_error'):
                print(f"   Take Profit Error: {order_info.get('take_profit_error')}")
        
        results["service"] = service_result
        
        # Wait a bit between orders
        await asyncio.sleep(2)
        
        # ========== Test 3: Environment Layer ==========
        print("\n" + "-" * 80)
        print("🎮 Test 3: Environment Layer (HyperliquidEnvironment.step)")
        print("-" * 80)
        
        env = ecp.get("hyperliquid")
        
        print(f"Creating order via Environment: LONG {qty} {symbol} with TP/SL")
        
        env_result = await env.step(
            symbol=symbol,
            action="LONG",
            qty=qty,
            leverage=10,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
        
        print(f"✅ Environment order result:")
        print(f"   Success: {env_result.get('success', 'N/A')}")
        print(f"   Message: {env_result.get('message', 'N/A')}")
        if 'extra' in env_result:
            extra = env_result['extra']
            if 'order_info' in extra:
                order_info = extra['order_info']
                print(f"   Order ID: {order_info.get('order_id', 'N/A')}")
                print(f"   Order Status: {order_info.get('order_status', 'N/A')}")
                if order_info.get('stop_loss_order'):
                    print(f"   Stop Loss Order: Created")
                if order_info.get('take_profit_order'):
                    print(f"   Take Profit Order: Created")
                if order_info.get('stop_loss_error'):
                    print(f"   Stop Loss Error: {order_info.get('stop_loss_error')}")
                if order_info.get('take_profit_error'):
                    print(f"   Take Profit Error: {order_info.get('take_profit_error')}")
        
        results["environment"] = env_result
        
        # ========== Summary ==========
        print("\n" + "=" * 80)
        print("📊 Summary")
        print("=" * 80)
        
        print("\n✅ All three layers tested successfully!")
        print("\nKey Observations:")
        print("1. Client Layer: Direct SDK call, returns raw SDK response")
        print("2. Service Layer: Wraps client, returns ActionResult with parsed order info")
        print("3. Environment Layer: Highest level, uses step() method with action='LONG'")
        
        # Check if main orders were created
        print("\nMain Order Status:")
        if results["client"] and 'main_order' in results["client"]:
            client_main = results["client"]["main_order"]
            if client_main.get('status') == 'ok':
                print("   ✅ Client: Main order created")
            else:
                print("   ❌ Client: Main order failed")
        else:
            print("   ⚠️  Client: No main_order in result")
        
        if results["service"] and results["service"].success:
            print("   ✅ Service: Main order created")
        else:
            print("   ❌ Service: Main order failed")
        
        if results["environment"] and results["environment"].get('success'):
            print("   ✅ Environment: Main order created")
        else:
            print("   ❌ Environment: Main order failed")
        
        # Cleanup
        await env.cleanup()
        await service.cleanup()
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
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
    base_dir = Path(root) / "workdir" / "online_trading_agent"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Test account (read-only, no private key needed for data stream)
    accounts = get_env("HYPERLIQUID_ACCOUNTS").get_secret_value()
    if accounts:
        accounts = json.loads(accounts)
    else:
        raise ValueError("HYPERLIQUID_ACCOUNTS not found")
    
    # Symbols to test
    test_symbols = ["BTC"]
    
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
            auto_start_data_stream=True,  # We'll start it manually
            symbol=test_symbols,
            data_type="candle"
        )
        
        await service.initialize()
        print("✅ Service initialized successfully")
        
        # Wait for data collection (need at least 60+ seconds for complete candle)
        wait_time = 70
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
                if symbol in data and "candles" in data[symbol]:
                    candles = data[symbol]["candles"]
                    print(f"\n✅ {symbol}: Found {len(candles)} candle(s)")
                    
                    # Show latest candle
                    if len(candles) > 0:
                        
                        for candle in candles:
                            latest = candle
                            print(f"   Latest candle:")
                            print(f"      Timestamp (ms): {latest.get('timestamp')}")
                            print(f"      Timestamp (UTC): {latest.get('timestamp_utc')}")
                            print(f"      Timestamp (Local): {latest.get('timestamp_local')}")
                            print(f"      Open Time (ms): {latest.get('open_time')}")
                            print(f"      Open Time (UTC): {latest.get('open_time_utc')}")
                            print(f"      Open Time (Local): {latest.get('open_time_local')}")
                            print(f"      Close Time (ms): {latest.get('close_time')}")
                            print(f"      Close Time (UTC): {latest.get('close_time_utc')}")
                            print(f"      Close Time (Local): {latest.get('close_time_local')}")
                            print(f"      Open: {latest.get('open')}")
                            print(f"      High: {latest.get('high')}")
                            print(f"      Low: {latest.get('low')}")
                            print(f"      Close: {latest.get('close')}")
                            print(f"      Volume: {latest.get('volume')}")
                            print(f"      Trade Count: {latest.get('trade_count')}")
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

async def test_database():
    """Print database contents from online_trading_agent database."""
    print("\n" + "=" * 60)
    print("Database Contents")
    print("=" * 60)
    
    # Database path: workdir/online_trading_agent/database/database.db
    database_dir = Path(root) / "workdir" / "online_trading_agent" / "database"
    database_service = DatabaseService(database_dir)
    
    try:
        # Connect to database
        await database_service.connect()
        print(f"✅ Connected to database: {database_dir / 'database.db'}")
        print()
        
        # Get all tables
        tables_result = await database_service.get_tables()
        if not tables_result.success:
            print(f"❌ Failed to get tables: {tables_result.message}")
            return
        
        tables_info = tables_result.extra.get("tables", [])
        total_tables = tables_result.extra.get("total_tables", 0)
        
        print(f"📊 Found {total_tables} table(s):")
        print()
        
        # Print each table's contents
        for table_info in tables_info:
            table_name = table_info["name"]
            row_count = table_info.get("row_count", 0)
            columns = table_info.get("columns", [])
            
            print(f"{'=' * 60}")
            print(f"Table: {table_name}")
            print(f"Rows: {row_count}")
            print(f"{'=' * 60}")
            
            if row_count == 0:
                print("(No data)")
                print()
                continue
            
            # Query all data from table first to get column names
            select_request = SelectRequest(
                table_name=table_name,
                order_by="timestamp DESC",
                limit=100  # Limit to 100 rows for display
            )
            
            result = await database_service.select_data(select_request)
            
            if result.success:
                data = result.extra.get("data", [])
                if data:
                    # Print each row as JSON string
                    for row in data:
                        print(json.dumps(row, ensure_ascii=False, indent=4))
                        exit()
                    
                    if row_count > 100:
                        print(f"\n... (showing first 100 of {row_count} rows)")
                else:
                    print("(No data returned)")
            else:
                print(f"❌ Failed to query table: {result.message}")
            
            print()
        
        print("=" * 60)
        print("✅ Database contents printed successfully")
        
    except Exception as e:
        print(f"❌ Error accessing database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Disconnect
        await database_service.disconnect()
        print("🔌 Disconnected from database")
        

async def test_snapshot():
    """Test snapshot data from Hyperliquid."""
    print("\n" + "=" * 60)
    print("Snapshot Data")
    print("=" * 60)

    # 初始化（默认会自动开启 websocket）
    info = Info()

    # 回调函数：当有新 K 线时自动触发
    def on_candle(data):
        print("收到 K 线数据：", data)

    # 订阅 BTC 的 1 分钟 K 线
    subscription = {
        "type": "candle",
        "coin": "BTC",   # 键为名字，内部会自动转换成 coinId
        "interval": "1m"
    }

    sub_id = info.subscribe(subscription, on_candle)
    print("订阅成功，subscription_id:", sub_id)


async def async_main():
    """Run async tests."""
    print("=" * 60)
    print("Hyperliquid Client Test Suite (Async)")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    
    # Test client basic functionality
    await test_client()
    
    # Test order creation
    # await test_orders()
    
    # Test complete data flow
    # await test_complete_data_flow()
    
    # Test database
    # await test_database()
    
    # Test snapshot
    # await test_snapshot()


async def main():
    
    args = parse_args()
    
    # Initialize configuration
    config.init_config(args.config, args)
    logger.init_logger(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize Hyperliquid service
    logger.info("| 🔧 Initializing Hyperliquid service...")
    accounts = get_env("HYPERLIQUID_ACCOUNTS").get_secret_value()
    if accounts:
        accounts = json.loads(accounts)
    config.hyperliquid_service.update(dict(accounts=accounts))
    hyperliquid_service = HyperliquidService(**config.hyperliquid_service)
    await hyperliquid_service.initialize()
    for env_name in config.env_names:
        env_config = config.get(f"{env_name}_environment", None)
        env_config.update(dict(hyperliquid_service=hyperliquid_service))
    logger.info(f"| ✅ Hyperliquid service initialized.")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")
    
    await async_main()
    
    logger.info("| 🚪 Test completed")

if __name__ == "__main__":
    asyncio.run(main())
