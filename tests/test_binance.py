"""Test script for Binance Spot and Futures WebSocket and REST API clients."""
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

load_dotenv()

from src.environments.binanceentry.spot_websocket import BinanceSpotWebSocket
from src.environments.binanceentry.futures_websocket import BinanceFuturesWebSocket
from src.environments.binanceentry.spot_client import BinanceSpotClient
from src.environments.binanceentry.futures_client import BinanceFuturesClient


def test_spot_websocket():
    """Test Binance Spot WebSocket."""
    print("=" * 60)
    print("Testing Binance Spot WebSocket (1m closed klines only)")
    print("=" * 60)
    
    message_count = 0
    should_stop = False
    
    def on_message(ws, data):
        nonlocal message_count, should_stop
        message_count += 1
        
        # Data is already processed by WebSocket class (only closed 1m klines)
        # timestamp is already formatted as 'YYYY-MM-DD HH:MM:SS'
        print(f"[Spot] {data['timestamp']} | {data['open_time']} | {data['close_time']} | {data['symbol']} | "
              f"Open: {data['open']} | Close: {data['close']} | "
              f"High: {data['high']} | Low: {data['low']} | "
              f"Volume: {data['volume']} | Interval: {data['interval']}")
        
        # Stop after receiving 5 closed klines
        if message_count >= 5:
            should_stop = True
            try:
                ws.close()
            except:
                pass
    
    def on_error(ws, error):
        print(f"[Spot] Error: {error}")
    
    def on_close(ws):
        print("[Spot] WebSocket closed")
    
    def on_open(ws):
        print("[Spot] WebSocket opened")
    
    # Create and start Spot WebSocket
    ws_client = BinanceSpotWebSocket(
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
        testnet=False
    )
    
    # Subscribe to 1-minute klines (only closed klines will be processed)
    ws_client.subscribe_kline("BTCUSDT", "1m")
    
    print("Starting Spot WebSocket...")
    ws_client.start()
    
    # Wait for messages
    try:
        while ws_client.is_running() and not should_stop:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Spot] Interrupted by user")
    finally:
        if not should_stop:
            ws_client.stop()
        print(f"[Spot] Test completed - received {message_count} closed klines\n")


def test_futures_websocket():
    """Test Binance Futures WebSocket."""
    print("=" * 60)
    print("Testing Binance Futures WebSocket (1m closed klines only)")
    print("=" * 60)
    
    message_count = 0
    should_stop = False
    
    def on_message(ws, data):
        nonlocal message_count, should_stop
        message_count += 1
        
        # Data is already processed by WebSocket class (only closed 1m klines)
        # timestamp is already formatted as 'YYYY-MM-DD HH:MM:SS'
        print(f"[Futures] {data['timestamp']} | {data['open_time']} | {data['close_time']} | {data['symbol']} | "
              f"Open: {data['open']} | Close: {data['close']} | "
              f"High: {data['high']} | Low: {data['low']} | "
              f"Volume: {data['volume']} | Interval: {data['interval']}")
        
        # Stop after receiving 5 closed klines
        if message_count >= 5:
            should_stop = True
            try:
                ws.close()
            except:
                pass
    
    def on_error(ws, error):
        print(f"[Futures] Error: {error}")
    
    def on_close(ws):
        print("[Futures] WebSocket closed")
    
    def on_open(ws):
        print("[Futures] WebSocket opened")
    
    # Create and start Futures WebSocket
    ws_client = BinanceFuturesWebSocket(
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
        testnet=False
    )
    
    # Subscribe to 1-minute klines (only closed klines will be processed)
    ws_client.subscribe_kline("ETHUSDT", "1m")
    
    print("Starting Futures WebSocket...")
    ws_client.start()
    
    # Wait for messages
    try:
        while ws_client.is_running() and not should_stop:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Futures] Interrupted by user")
    finally:
        if not should_stop:
            ws_client.stop()
        print(f"[Futures] Test completed - received {message_count} closed klines\n")


def test_spot_client():
    """Test Binance Spot REST API client."""
    print("=" * 60)
    print("Testing Binance Spot REST API Client")
    print("=" * 60)
    
    # Get API credentials from environment
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_SECRET_KEY", "")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    
    if not api_key or not api_secret:
        print("⚠️  BINANCE_API_KEY and BINANCE_SECRET_KEY not found in environment variables")
        print("   Skipping Spot Client tests...\n")
        return
    
    try:
        # Initialize Spot client
        spot_client = BinanceSpotClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        
        print(f"✅ Spot Client initialized (testnet={testnet})")
        print(f"   Base URL: {spot_client.base_url}\n")
        
        # Test 1: Get exchange info (no authentication required)
        print("Test 1: Getting exchange info (public endpoint)...")
        try:
            exchange_info = spot_client.exchange_info()
            symbols_count = len(exchange_info.get('symbols', []))
            print(f"✅ Exchange info retrieved successfully")
            print(f"   Total symbols: {symbols_count}")
            if symbols_count > 0:
                sample_symbol = exchange_info['symbols'][0]
                print(f"   Sample symbol: {sample_symbol.get('symbol')} ({sample_symbol.get('status')})")
        except Exception as e:
            print(f"❌ Failed to get exchange info: {e}\n")
            return
        
        print()
        
        # Test 2: Get account info (authentication required)
        print("Test 2: Getting account info (authenticated endpoint)...")
        try:
            account_info = spot_client.account()
            print(f"✅ Account info retrieved successfully")
            print(f"   Account type: {account_info.get('accountType')}")
            print(f"   Permissions: {account_info.get('permissions', [])}")
            balances = account_info.get('balances', [])
            non_zero_balances = [b for b in balances if float(b.get('free', 0)) + float(b.get('locked', 0)) > 0]
            print(f"   Non-zero balances: {len(non_zero_balances)}")
            if non_zero_balances:
                for balance in non_zero_balances[:3]:  # Show first 3
                    asset = balance.get('asset')
                    free = balance.get('free')
                    locked = balance.get('locked')
                    print(f"     {asset}: free={free}, locked={locked}")
        except Exception as e:
            print(f"❌ Failed to get account info: {e}")
            if "401" in str(e) or "Invalid" in str(e):
                print("   This might be due to:")
                print("   - Invalid API key or secret key")
                print("   - API key not enabled for Spot trading")
                print("   - IP whitelist restrictions")
        
        print()
        
    except Exception as e:
        print(f"❌ Error initializing Spot Client: {e}\n")


def test_futures_client():
    """Test Binance Futures REST API client."""
    print("=" * 60)
    print("Testing Binance Futures REST API Client")
    print("=" * 60)
    
    # Get API credentials from environment
    api_key = os.getenv("BINANCE_TESTNET_TRADING_API_KEY", "")
    api_secret = os.getenv("BINANCE_TESTNET_TRADING_SECRET_KEY", "")
    testnet = True
    
    if not api_key or not api_secret:
        print("⚠️  BINANCE_TESTNET_TRADING_API_KEY and BINANCE_TESTNET_TRADING_SECRET_KEY not found in environment variables")
        print("   Skipping Futures Client tests...\n")
        return
    
    try:
        # Initialize Futures client
        futures_client = BinanceFuturesClient(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )
        
        print(f"✅ Futures Client initialized (testnet={testnet})")
        print(f"   Base URL: {futures_client.base_url}\n")
        
        # Test 1: Get account info (authentication required)
        print("Test 1: Getting futures account info (authenticated endpoint)...")
        try:
            account_info = futures_client.get_account()
            print(f"✅ Futures account info retrieved successfully")
            print(f"   Total wallet balance: {account_info.get('totalWalletBalance')}")
            print(f"   Available balance: {account_info.get('availableBalance')}")
            print(f"   Total unrealized profit: {account_info.get('totalUnrealizedProfit')}")
            assets = account_info.get('assets', [])
            non_zero_assets = [a for a in assets if float(a.get('walletBalance', 0)) > 0]
            print(f"   Non-zero assets: {len(non_zero_assets)}")
            if non_zero_assets:
                for asset in non_zero_assets[:3]:  # Show first 3
                    asset_name = asset.get('asset')
                    wallet_balance = asset.get('walletBalance')
                    print(f"     {asset_name}: walletBalance={wallet_balance}")
        except Exception as e:
            print(f"❌ Failed to get futures account info: {e}")
            if "401" in str(e) or "Invalid" in str(e):
                print("   This might be due to:")
                print("   - Invalid API key or secret key")
                print("   - API key not enabled for Futures trading")
                print("   - IP whitelist restrictions")
        
        print()
        
        # Test 2: Get position risk
        print("Test 2: Getting position risk (authenticated endpoint)...")
        try:
            positions = futures_client.get_position_risk()
            print(f"✅ Position risk retrieved successfully")
            print(f"   Total positions: {len(positions)}")
            open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            print(f"   Open positions: {len(open_positions)}")
            if open_positions:
                for position in open_positions[:3]:  # Show first 3
                    symbol = position.get('symbol')
                    position_amt = position.get('positionAmt')
                    entry_price = position.get('entryPrice')
                    unrealized_profit = position.get('unRealizedProfit')
                    print(f"     {symbol}: amount={position_amt}, entry={entry_price}, PnL={unrealized_profit}")
            else:
                print("   No open positions")
        except Exception as e:
            print(f"❌ Failed to get position risk: {e}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error initializing Futures Client: {e}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Binance Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Spot REST API Client
    # test_spot_client()
    # time.sleep(1)
    
    # Test 2: Futures REST API Client
    test_futures_client()
    time.sleep(1)
    
    # # Test 3: Spot WebSocket
    # print("\n" + "=" * 60)
    # print("WebSocket Tests")
    # print("=" * 60 + "\n")
    # test_spot_websocket()
    # time.sleep(2)
    
    # # Test 4: Futures WebSocket
    # test_futures_websocket()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
