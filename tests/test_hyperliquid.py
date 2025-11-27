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
from typing import Any

from hyperliquid.info import Info

import argparse
from mmengine import DictAction


root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

load_dotenv()

from src.environments.hyperliquidentry.client import HyperliquidClient
from src.environments.hyperliquidentry.service import OnlineHyperliquidService
from src.environments.hyperliquidentry.service import OfflineHyperliquidService
from src.environments.hyperliquidentry.types import (
    GetDataRequest, 
    CreateOrderRequest, 
    OrderType, 
    GetPositionsRequest,
    GetOrdersRequest,
    GetAccountRequest, 
    GetExchangeInfoRequest,
    CloseOrderRequest,
)
from src.logger import logger
from src.config import config
from src.utils import get_env
from src.utils import dedent
from src.environments import ecp
from src.utils import get_standard_timestamp
from src.environments.database.service import DatabaseService
from src.environments.database.types import QueryRequest, SelectRequest

async def test_online_hyperliquid():
    
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
    hyperliquid_service = OnlineHyperliquidService(**config.hyperliquid_service)
    await hyperliquid_service.initialize()
    for env_name in config.env_names:
        env_config = config.get(f"{env_name}_environment", None)
        env_config.update(dict(hyperliquid_service=hyperliquid_service))
    logger.info(f"| ✅ Hyperliquid service initialized.")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")


def _safe_float_format(value: Any, default: float = 0.0) -> str:
    """Safely format float value (simulates HyperliquidEnvironment)."""
    try:
        if value is None:
            return f"{float(default):.2f}"
        if isinstance(value, (int, float)):
            return f"{float(value):.2f}"
        if isinstance(value, str):
            return f"{float(value):.2f}"
        else:
            return f"{float(default):.2f}"
    except (ValueError, TypeError):
        return f"{float(default):.2f}"

async def test_offline_hyperliquid():
    def parse_args():
        parser = argparse.ArgumentParser(description='Offline Trading Agent Example')
        parser.add_argument("--config", default=os.path.join(root, "configs", "offline_trading_agent.py"), help="config file path")
        
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
    hyperliquid_service = OfflineHyperliquidService(**config.hyperliquid_service)
    await hyperliquid_service.initialize()
    for env_name in config.env_names:
        env_config = config.get(f"{env_name}_environment", None)
        env_config.update(dict(hyperliquid_service=hyperliquid_service))
    logger.info(f"| ✅ Hyperliquid service initialized.")
    
    # Initialize environments
    logger.info("| 🎮 Initializing environments...")
    await ecp.initialize(config.env_names)
    logger.info(f"| ✅ Environments initialized: {ecp.list()}")
    
    # Test Hyperliquid service
    logger.info("| 🧪 Testing Hyperliquid service...")
    # 1. Get exchange info
    exchange_info= await hyperliquid_service.get_exchange_info(GetExchangeInfoRequest())
    logger.info(f"| 📝 Exchange info: {exchange_info}")
    
    # 2. Get account
    account= await hyperliquid_service.get_account(GetAccountRequest(account_name="account1"))
    logger.info(f"| 📝 Account: {account}")
    
    # 3. Get positions
    positions= await hyperliquid_service.get_positions(GetPositionsRequest(account_name="account1"))
    logger.info(f"| 📝 Positions: {positions}")
    
    # 4. Get orders
    orders= await hyperliquid_service.get_orders(GetOrdersRequest(account_name="account1"))
    logger.info(f"| 📝 Orders: {orders}")
    
    # 5. Create order
    order= await hyperliquid_service.create_order(CreateOrderRequest(
            account_name="account1", 
            symbol="BTC", 
            side="buy", 
            qty=0.001,
            stop_loss_price=100000,
            take_profit_price=110000,
        )
    )
    logger.info(f"| 📝 Created Order: {order}")
    positions= await hyperliquid_service.get_positions(GetPositionsRequest(account_name="account1"))
    logger.info(f"| 📝 Positions: {positions}")
    orders= await hyperliquid_service.get_orders(GetOrdersRequest(account_name="account1"))
    logger.info(f"| 📝 Orders: {orders}")
    
    # 6. Close Order
    close_order= await hyperliquid_service.close_order(CloseOrderRequest(
        account_name="account1", 
        symbol="BTC", 
        side="sell"
    ))
    logger.info(f"| 📝 Closed Order: {close_order}")
    positions= await hyperliquid_service.get_positions(GetPositionsRequest(account_name="account1"))
    logger.info(f"| 📝 Positions: {positions}")
    orders= await hyperliquid_service.get_orders(GetOrdersRequest(account_name="account1"))
    logger.info(f"| 📝 Orders: {orders}")
    
    # 7. Get data
    for i in range(100):
        data_result = await hyperliquid_service.get_data(GetDataRequest(
            account_name="account1", 
            symbol="BTC", 
            interval="1m",
            limit=30
        ))
        data_result_extra = data_result.extra
        
        candles = {}
        indicators = {}
        for symbol, data in data_result_extra.get("data", {}).items():
            candles[symbol] = data.get("candles", data.get("candle", []))  # Support both formats
            indicators[symbol] = data.get("indicators", [])  # Indicator data read from database
        
        candles_string = ""
        for symbol, candles_list in candles.items():
            if not candles_list:
                continue
            
            # Create table (fully simulates HyperliquidEnvironment format)
            symbol_string = f"Symbol: {symbol}. History {len(candles_list)} minutes candles data.\n"
            symbol_string += "| Timestamp           | Open | High | Low | Close | Volume | Trade Count |\n"
            symbol_string += "|---------------------|------|------|-----|-------|--------|-------------|\n"
            
            # Add table rows
            for candle in candles_list:
                timestamp = candle.get("timestamp_local", candle.get("timestamp_utc", ""))
                open_val = _safe_float_format(candle.get("open"))
                high_val = _safe_float_format(candle.get("high"))
                low_val = _safe_float_format(candle.get("low"))
                close_val = _safe_float_format(candle.get("close"))
                volume_val = _safe_float_format(candle.get("volume"))
                trade_count_val = _safe_float_format(candle.get("trade_count", 0))
                symbol_string += f"| {timestamp:<19} | {open_val:>10} | {high_val:>10} | {low_val:>10} | {close_val:>10} | {volume_val:>10} | {trade_count_val:>11} |\n"
            
            candles_string += symbol_string + "\n"
        
        indicators_string = ""
        for symbol, indicators_list in indicators.items():
            if not indicators_list:
                continue
            
            # Get indicator names
            indicator_names = hyperliquid_service.indicators_name
            if not indicator_names:
                continue
            
            # Create table header
            symbol_string = f"Symbol: {symbol}. History {len(indicators_list)} minutes indicators data.\n"
            # Build header row: Timestamp | Timestamp (Local) | indicator1 | indicator2 | ...
            header_cols = ["Timestamp           "] + [name.upper() for name in indicator_names]
            symbol_string += "| " + " | ".join(header_cols) + " |\n"
            
            # Build separator row
            separator_cols = ["---------------------"] + ["-" * max(12, len(name.upper())) for name in indicator_names]
            separator = "| " + " | ".join(separator_cols) + " |\n"
            symbol_string += separator
            
            # Add table rows
            for indicator in indicators_list:
                timestamp = indicator.get("timestamp_local", "")
                row_values = [str(timestamp)]
                
                # Add indicator values in the same order as header
                for indicator_name in indicator_names:
                    indicator_value = indicator.get(indicator_name, None)
                    if indicator_value is not None:
                        row_values.append(_safe_float_format(indicator_value))
                    else:
                        row_values.append("")
                
                # Format row with alignment
                formatted_row = f"| {row_values[0]:<19} |"
                for val in row_values[1:]:
                    formatted_row += f" {val:>12} |"
                formatted_row += "\n"
                symbol_string += formatted_row
            
        indicators_string += symbol_string + "\n"
        
        data_string = dedent(f"""
            <data>
            <candles>
            {candles_string}
            </candles>
            <indicators>
            {indicators_string}
            </indicators>
            </data>
        """)
        logger.info(f"| 📝 Data: {data_string}")

if __name__ == "__main__":
    asyncio.run(test_offline_hyperliquid())
