"""Test script for Hyperliquid REST API client."""
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

load_dotenv()

from src.environments.hyperliquidentry.client import HyperliquidClient
from src.logger import logger


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
        
        # Show some account details
        if isinstance(account, dict):
            for key in ['marginSummary', 'assetPositions', 'withdrawable']:
                if key in account:
                    print(f"  {key}: {account[key]}")
        
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
        testnet=True  # Use testnet for testing
    )
    
    try:
        # Test market order
        print("Creating market order: BUY 0.01 BTC")
        result = client.create_order(
            symbol="BTC",
            side="buy",  # Will be converted to boolean
            order_type="Market",
            size=0.01,
            reduce_only=False
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


def main():
    """Run all tests."""
    print("=" * 60)
    print("Hyperliquid Client Test Suite")
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
    
    print("\n" + "=" * 60)
    print("Test Suite Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()

