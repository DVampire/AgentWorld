"""Test script for fetching Hyperliquid minute-level data and storing in database with indicators."""

import asyncio
import time
from pathlib import Path
import sys

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from hyperliquid.info import Info
from mmengine import DictAction
import argparse
import os

from src.config import config
from src.environment.database.service import DatabaseService
from src.environment.database.types import CreateTableRequest, InsertRequest, QueryRequest, SelectRequest
from src.environment.hyperliquidentry.exceptions import HyperliquidError
from src.registry import INDICATOR
from src.logger import logger
from src.utils.calender_utils import get_standard_timestamp
import pandas as pd
from typing import Optional, List, Dict, Union


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


class CandleHandlerNoCacheLimit:
    """Handler for candle (OHLCV) data with streaming, caching, and technical indicators.
    
    This version removes cache limit to calculate indicators for all data.
    """
    
    def __init__(self, database_service: DatabaseService):
        """Initialize candle handler.
        
        Args:
            database_service: Database service instance
        """
        self.database_service = database_service
        
        # Cache for candle data (for fast indicator calculation)
        # Key: symbol, Value: DataFrame with recent candles (NO LIMIT)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_limit: Optional[int] = None  # No limit - use all data
        
        self._indicators_functions = {}
        self._indicators_name = []
        for indicator in config.hyperliquid_indicators:
            indicator_function = INDICATOR.build(cfg=dict(type=indicator))
            self._indicators_functions[indicator] = indicator_function
            self._indicators_name.extend(indicator_function.indicators_name)
        
        # number of concurrent coroutines
        self._concurrent_coroutines = 8
    
    def _sanitize_table_name(self, symbol: str) -> str:
        """Sanitize symbol name to be used as table name."""
        # Replace invalid characters with underscore
        table_name = symbol.replace("/", "_").replace(".", "_").replace("-", "_")
        # Remove any other invalid characters
        table_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)
        return f"data_{table_name}"
    
    async def ensure_table_exists(self, symbol: str) -> None:
        """Ensure candle table and indicators table exist for a symbol.
        
        Args:
            symbol: Symbol name (should be uppercase for consistency)
        """
        # Normalize symbol to uppercase for consistency
        symbol = symbol.upper() if symbol else ""
        base_name = self._sanitize_table_name(symbol)
        candle_table_name = f"{base_name}_candle"
        indicators_table_name = f"{base_name}_indicators"
        
        # Ensure candle table exists
        check_candle_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{candle_table_name}'"
        check_candle_result = await self.database_service.execute_query(
            QueryRequest(query=check_candle_query)
        )
        
        if not (check_candle_result.success and check_candle_result.extra.get("data")):
            # Create candle table
            columns = [
                {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
                {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "interval", "type": "TEXT"},  # e.g., "1m", "5m", "1h", "1d"
                {"name": "timestamp", "type": "INTEGER", "constraints": "NOT NULL"},
                {"name": "timestamp_utc", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "timestamp_local", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "open_time", "type": "INTEGER", "constraints": "NOT NULL"},
                {"name": "open_time_utc", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "open_time_local", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "close_time", "type": "INTEGER", "constraints": "NOT NULL"},
                {"name": "close_time_utc", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "close_time_local", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "open", "type": "REAL"},
                {"name": "high", "type": "REAL"},
                {"name": "low", "type": "REAL"},
                {"name": "close", "type": "REAL"},
                {"name": "volume", "type": "REAL"},
                {"name": "trade_count", "type": "REAL"},
                {"name": "is_closed", "type": "INTEGER"},  # 0 or 1 (boolean)
                {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
            ]
            
            create_request = CreateTableRequest(
                table_name=candle_table_name,
                columns=columns,
                primary_key=None  # Primary key is already in the id column constraints
            )
            result = await self.database_service.create_table(create_request)
            if not result.success:
                logger.error(f"Failed to create candle table {candle_table_name}: {result.message}")
                raise HyperliquidError(f"Failed to create candle table {candle_table_name}: {result.message}")
            
            # Create unique constraint to prevent duplicate entries for same symbol and timestamp
            unique_constraint_name = f"{candle_table_name}_unique_time"
            unique_query = f"CREATE UNIQUE INDEX IF NOT EXISTS {unique_constraint_name} ON {candle_table_name}(symbol, timestamp)"
            unique_result = await self.database_service.execute_query(QueryRequest(query=unique_query))
            if not unique_result.success:
                logger.warning(f"Failed to create unique constraint {unique_constraint_name}: {unique_result.message}")
            
            # Create index for performance optimization (only on timestamp)
            # Using ASC to match common query patterns (historical data in chronological order)
            index_name = f"{candle_table_name}_timestamp_idx"
            index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {candle_table_name}(timestamp ASC)"
            index_result = await self.database_service.execute_query(QueryRequest(query=index_query))
            if not index_result.success:
                logger.warning(f"Failed to create index {index_name} for {candle_table_name}: {index_result.message}")
        
        # Ensure indicators table exists
        check_indicators_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{indicators_table_name}'"
        check_indicators_result = await self.database_service.execute_query(
            QueryRequest(query=check_indicators_query)
        )
        
        if not (check_indicators_result.success and check_indicators_result.extra.get("data")):
            # Create indicators table with all technical indicators
            columns = [
                {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
                {"name": "timestamp", "type": "INTEGER", "constraints": "NOT NULL"},
                {"name": "timestamp_utc", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "timestamp_local", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
            ]
            # Add indicator columns dynamically from self._indicators_name
            for indicator_name in self._indicators_name:
                columns.append({"name": indicator_name, "type": "REAL"})
            columns.append({"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"})
            
            create_request = CreateTableRequest(
                table_name=indicators_table_name,
                columns=columns,
                primary_key=None
            )
            result = await self.database_service.create_table(create_request)
            if not result.success:
                logger.error(f"Failed to create indicators table {indicators_table_name}: {result.message}")
                raise HyperliquidError(f"Failed to create indicators table {indicators_table_name}: {result.message}")
            
            # Create unique constraint to prevent duplicate entries for same symbol and timestamp
            unique_constraint_name = f"{indicators_table_name}_unique_time"
            unique_query = f"CREATE UNIQUE INDEX IF NOT EXISTS {unique_constraint_name} ON {indicators_table_name}(symbol, timestamp)"
            unique_result = await self.database_service.execute_query(QueryRequest(query=unique_query))
            if not unique_result.success:
                logger.warning(f"Failed to create unique constraint {unique_constraint_name}: {unique_result.message}")
            
            # Create index for performance optimization
            # Using ASC to match common query patterns (historical data in chronological order)
            index_name = f"{indicators_table_name}_timestamp_idx"
            index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {indicators_table_name}(timestamp ASC)"
            index_result = await self.database_service.execute_query(QueryRequest(query=index_query))
            if not index_result.success:
                logger.warning(f"Failed to create index {index_name} for {indicators_table_name}: {index_result.message}")
    
    async def get_indicators_name(self) -> List[str]:
        """Get indicators name."""
        return self._indicators_name
    
    async def _preprocess_data(self, data: Dict, symbol: str) -> Dict:
        """Preprocess data for insertion."""
        symbol = symbol.upper() if symbol else data.get("s", "").upper()
        interval = data.get("i", "1m")
        open_time = int(data.get("t", 0))
        close_time = int(data.get("T", 0))
        open_price = float(data.get("o", 0))
        high_price = float(data.get("h", 0))
        low_price = float(data.get("l", 0))
        close_price = float(data.get("c", 0))
        volume = float(data.get("v", 0))
        trade_count = float(data.get("n", 0))
        
        timestamp_dict = get_standard_timestamp(close_time + 1000)
        timestamp = timestamp_dict["timestamp"]
        timestamp_utc = timestamp_dict["timestamp_utc"]
        timestamp_local = timestamp_dict["timestamp_local"]
        
        open_time_dict = get_standard_timestamp(open_time)
        open_time_utc = open_time_dict["timestamp_utc"]
        open_time_local = open_time_dict["timestamp_local"]
        
        close_time_dict = get_standard_timestamp(close_time)
        close_time_utc = close_time_dict["timestamp_utc"]
        close_time_local = close_time_dict["timestamp_local"]
        
        current_timestamp = int(time.time()) * 1000
        is_closed = 1 if current_timestamp > close_time else 0
        
        res = {
            "symbol": symbol,
            "interval": interval,
            "timestamp": timestamp,
            "timestamp_utc": timestamp_utc,
            "timestamp_local": timestamp_local,
            "open_time": open_time,
            "open_time_utc": open_time_utc,
            "open_time_local": open_time_local,
            "close_time": close_time,
            "close_time_utc": close_time_utc,
            "close_time_local": close_time_local,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
            "trade_count": trade_count,
            "is_closed": is_closed,
        }
        
        return res
    
    async def _calculate_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate indicators for a given dataframe."""
        all_indicators_df = pd.DataFrame(index=df.index)
        
        indicators_list = [(indicator_obj.indicators_name, indicator_obj) 
                          for _, indicator_obj in self._indicators_functions.items()]
        
        for i in range(0, len(indicators_list), self._concurrent_coroutines):
            batch = indicators_list[i:i+self._concurrent_coroutines]
            
            indicator_tasks = []
            indicator_names = []
            for indicator_name, indicator_obj in batch:
                indicator_tasks.append(indicator_obj(df.copy()))
                indicator_names.append(indicator_name)
            
            indicator_results = await asyncio.gather(*indicator_tasks, return_exceptions=True)
            
            for indicator_name, result in zip(indicator_names, indicator_results):
                if isinstance(result, Exception):
                    logger.warning(f"| ⚠️  Error calculating indicator {indicator_name} for {symbol}: {result}")
                    continue
                if isinstance(result, pd.DataFrame) and not result.empty:
                    for col in result.columns:
                        all_indicators_df[col] = result[col]
        
        return all_indicators_df
    
    async def full_insert(self, data: List[Dict], symbol: str) -> Dict:
        """Insert candle data from list of dictionaries with NO cache limit."""
        try:
            symbol = symbol.upper() if symbol else ""
            
            if not data:
                logger.warning(f"| ⚠️  No data provided for full_insert for {symbol}")
                return {"success": False, "message": "No data provided"}
            
            await self.ensure_table_exists(symbol)
            
            base_name = self._sanitize_table_name(symbol)
            candle_table_name = f"{base_name}_candle"
            indicators_table_name = f"{base_name}_indicators"
            
            # Step 1: Preprocess all data
            logger.info(f"| 📝 Preprocessing {len(data)} records for {symbol}")
            processed_data = []
            for i in range(0, len(data), self._concurrent_coroutines):
                batch = data[i:i+self._concurrent_coroutines]
                tasks = [self._preprocess_data(d, symbol) for d in batch]
                processed_data.extend(await asyncio.gather(*tasks))
            
            # Step 2: Insert all data into database
            logger.info(f"| 💾 Inserting {len(processed_data)} records into {candle_table_name}")
            
            if not processed_data:
                logger.warning(f"| ⚠️  No processed data to insert for {symbol}")
                return {"success": False, "message": "No processed data to insert"}
            
            insert_request = InsertRequest(
                table_name=candle_table_name,
                data=processed_data,
                on_conflict="REPLACE"
            )
            result = await self.database_service.insert_data(insert_request)
            
            if not result.success:
                logger.error(f"| ❌ Failed to insert candle data for {symbol}: {result.message}")
                return {"success": False, "message": f"Failed to insert candle data: {result.message}"}
            
            rowcount = result.extra.get("row_count", len(processed_data)) if result.extra else len(processed_data)
            logger.info(f"| ✅ Inserted {rowcount} candle records for {symbol}")
            
            # Step 3: Update cache (NO LIMIT)
            logger.info(f"| 📦 Updating cache for {symbol} (no limit)")
            await self._update_cache(symbol, processed_data)
            logger.info(f"| ✅ Cache updated for {symbol} (now has {len(self._cache[symbol])} records)")
            
            # Step 4: Calculate indicators using ALL processed_data (not just cache)
            logger.info(f"| 📊 Calculating indicators for {symbol} using {len(self._indicators_functions)} indicator(s) on {len(processed_data)} records")
            
            # Create DataFrame from ALL processed_data
            df = pd.DataFrame(processed_data)
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            all_indicators_df = await self._calculate_indicators(df, symbol)
            
            if all_indicators_df.empty or len(all_indicators_df.columns) == 0:
                logger.warning(f"| ⚠️  No indicators calculated for {symbol}")
                return {"success": True, "message": f"Inserted {len(processed_data)} records, but no indicators calculated"}
            
            # Step 5: Insert indicators for ALL records
            logger.info(f"| 💾 Inserting indicators for {len(df)} records into {indicators_table_name}")
            
            indicators_to_insert = []
            for idx, row in df.iterrows():
                indicator_data = {
                    "timestamp": int(row["timestamp"]),
                    "timestamp_utc": row.get("timestamp_utc", ""),
                    "timestamp_local": row.get("timestamp_local", ""),
                    "symbol": symbol,
                }
                
                if idx in all_indicators_df.index:
                    indicator_row = all_indicators_df.loc[idx]
                    for indicator_name in self._indicators_name:
                        if indicator_name in indicator_row.index:
                            value = indicator_row[indicator_name]
                            if pd.notna(value):
                                indicator_data[indicator_name] = float(value)
                            else:
                                indicator_data[indicator_name] = None
                
                indicators_to_insert.append(indicator_data)
            
            if indicators_to_insert:
                insert_request = InsertRequest(
                    table_name=indicators_table_name,
                    data=indicators_to_insert,
                    on_conflict="REPLACE"
                )
                result = await self.database_service.insert_data(insert_request)
                
                if result.success:
                    rowcount = result.extra.get("row_count", len(indicators_to_insert)) if result.extra else len(indicators_to_insert)
                    logger.info(f"| ✅ Inserted {rowcount} indicator records for {symbol}")
                else:
                    logger.warning(f"| ⚠️  Failed to insert indicators for {symbol}: {result.message}")
            
            return {
                "success": True,
                "message": f"Successfully inserted {len(processed_data)} candle records and {len(indicators_to_insert)} indicator records for {symbol}"
            }
            
        except Exception as e:
            logger.error(f"| ❌ Error in full_insert for {symbol}: {e}", exc_info=True)
            return {"success": False, "message": f"Error in full_insert: {str(e)}"}
    
    async def _update_cache(self, symbol: str, candle_data: Union[Dict, List[Dict]]) -> None:
        """Update cache with new candle data. NO LIMIT - stores all data."""
        if symbol not in self._cache:
            self._cache[symbol] = pd.DataFrame()
        
        if isinstance(candle_data, dict):
            new_data = pd.DataFrame([candle_data])
        else:
            new_data = pd.DataFrame(candle_data)
        
        df = self._cache[symbol]
        
        if df.empty:
            df = new_data.copy()
        else:
            df_combined = pd.concat([df, new_data], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
            if "timestamp" in df_combined.columns:
                df_combined = df_combined.sort_values("timestamp")
            df = df_combined
        
        # NO LIMIT - store all data
        self._cache[symbol] = df
    
    async def get_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, List[Dict]]:
        """Get candle data and indicators from database."""
        symbol = symbol.upper() if symbol else ""
        base_name = self._sanitize_table_name(symbol)
        candle_table_name = f"{base_name}_candle"
        indicators_table_name = f"{base_name}_indicators"
        
        if start_date and end_date:
            candle_request = SelectRequest(
                table_name=candle_table_name,
                where_clause="symbol = :symbol AND timestamp >= :start_date AND timestamp <= :end_date",
                where_params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
                order_by="timestamp ASC",
                limit=limit
            )
        else:
            if limit:
                candle_request = SelectRequest(
                    table_name=candle_table_name,
                    where_clause="symbol = :symbol",
                    where_params={"symbol": symbol},
                    order_by="timestamp DESC",
                    limit=limit
                )
            else:
                max_timestamp_query = f"SELECT MAX(timestamp) as max_ts FROM {candle_table_name} WHERE symbol = ?"
                max_ts_result = await self.database_service.execute_query(
                    QueryRequest(query=max_timestamp_query, parameters=(symbol,))
                )
                
                if not max_ts_result.success or not max_ts_result.extra.get("data"):
                    logger.warning(f"| ⚠️  Failed to get max timestamp for {symbol}: {max_ts_result.message}")
                    return {"candles": [], "indicators": []}
                
                max_ts_data = max_ts_result.extra.get("data", [])
                if not max_ts_data or not max_ts_data[0].get("max_ts"):
                    logger.warning(f"| ⚠️  No data found for {symbol}")
                    return {"candles": [], "indicators": []}
                
                latest_timestamp = max_ts_data[0]["max_ts"]
                candle_request = SelectRequest(
                    table_name=candle_table_name,
                    where_clause="symbol = :symbol AND timestamp = :timestamp",
                    where_params={"symbol": symbol, "timestamp": latest_timestamp},
                    order_by="timestamp ASC"
                )
        
        candle_result = await self.database_service.select_data(candle_request)
        
        if not candle_result.success:
            logger.warning(f"| ⚠️  Failed to query candles: {candle_result.message}")
            return {"candles": [], "indicators": []}
        
        candle_data = candle_result.extra.get("data", [])
        
        if limit and not start_date and not end_date:
            candle_data.reverse()
        
        if start_date and end_date:
            indicators_request = SelectRequest(
                table_name=indicators_table_name,
                where_clause="symbol = :symbol AND timestamp >= :start_date AND timestamp <= :end_date",
                where_params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
                order_by="timestamp ASC",
                limit=limit
            )
        else:
            if limit:
                indicators_request = SelectRequest(
                    table_name=indicators_table_name,
                    where_clause="symbol = :symbol",
                    where_params={"symbol": symbol},
                    order_by="timestamp DESC",
                    limit=limit
                )
            else:
                if candle_data:
                    latest_timestamp = candle_data[-1].get("timestamp") if candle_data else None
                    if latest_timestamp:
                        indicators_request = SelectRequest(
                            table_name=indicators_table_name,
                            where_clause="symbol = :symbol AND timestamp = :timestamp",
                            where_params={"symbol": symbol, "timestamp": latest_timestamp},
                            order_by="timestamp ASC"
                        )
                    else:
                        indicators_data = []
                        return {
                            "candles": candle_data,
                            "indicators": indicators_data
                        }
                else:
                    indicators_data = []
                    return {
                        "candles": candle_data,
                        "indicators": indicators_data
                    }
        
        indicators_result = await self.database_service.select_data(indicators_request)
        
        if indicators_result.success:
            indicators_data = indicators_result.extra.get("data", [])
            
            if limit and not start_date and not end_date:
                indicators_data.reverse()
        else:
            indicators_data = []
        
        return {
            "candles": candle_data,
            "indicators": indicators_data
        }


async def fetch_and_store_hyperliquid_data(
    symbol: str,
    start_time: int,
    end_time: int,
    interval: str = "1m",
    db_dir: str = "workdir/hyperliquid_data"
):
    """
    Fetch Hyperliquid minute-level data and store in database with indicators.
    
    Args:
        symbol: Trading symbol (e.g., "BTC", "ETH")
        start_time: Start time in format "YYYY-MM-DD HH:MM:SS" (local time)
        end_time: End time in format "YYYY-MM-DD HH:MM:SS" (local time)
        interval: Data interval (default: "1m" for 1 minute)
        db_dir: Database directory path (default: "workdir/hyperliquid_data")
    """
    try:
        # Step 1: Initialize Hyperliquid Info client
        logger.info(f"| 🔌 Initializing Hyperliquid API client")
        info = Info(base_url="https://api.hyperliquid.xyz")
        
        # Step 2: Convert local time strings to UTC timestamps (milliseconds)
        logger.info(f"| 📅 Converting time range: {start_time} to {end_time}")
        
        logger.info(f"| 📊 Start timestamp (UTC): {get_standard_timestamp(start_time)['timestamp_utc']} ({start_time})")
        logger.info(f"| 📊 End timestamp (UTC): {get_standard_timestamp(end_time)['timestamp_utc']} ({end_time})")
        
        # Step 3: Fetch data from Hyperliquid API
        logger.info(f"| 📥 Fetching {interval} data for {symbol} from Hyperliquid API...")
        symbol_data = info.candles_snapshot(symbol, interval, start_time, end_time)
        
        if not symbol_data:
            logger.warning(f"| ⚠️  No data returned from API for {symbol}")
            return
        
        logger.info(f"| ✅ Fetched {len(symbol_data)} candles from API")
        
        # Step 4: Initialize database service
        logger.info(f"| 🗄️  Initializing database service at {db_dir}")
        db_path = Path(db_dir)
        database_service = DatabaseService(db_path)
        await database_service.connect()
        logger.info(f"| ✅ Database connected at {db_path / 'database.db'}")
        
        # Step 5: Initialize CandleHandler (no cache limit version)
        logger.info(f"| 📊 Initializing CandleHandlerNoCacheLimit")
        candle_handler = CandleHandlerNoCacheLimit(database_service=database_service)
        
        # Step 6: Insert data and calculate indicators
        logger.info(f"| 💾 Inserting {len(symbol_data)} candles and calculating indicators...")
        result = await candle_handler.full_insert(symbol_data, symbol)
        
        if result.get("success"):
            logger.info(f"| ✅ Successfully inserted data: {result.get('message')}")
        else:
            logger.error(f"| ❌ Failed to insert data: {result.get('message')}")
            return
        
        # Step 7: Verify data insertion
        logger.info(f"| 🔍 Verifying data insertion...")
        
        # Query total count from database
        base_name = candle_handler._sanitize_table_name(symbol.upper())
        candle_table_name = f"{base_name}_candle"
        indicators_table_name = f"{base_name}_indicators"
        
        # Get total count of candles
        count_candle_query = f"SELECT COUNT(*) as count FROM {candle_table_name} WHERE symbol = ?"
        count_candle_result = await database_service.execute_query(
            QueryRequest(query=count_candle_query, parameters=(symbol.upper(),))
        )
        candles_count = 0
        if count_candle_result.success and count_candle_result.extra.get("data"):
            candles_count = count_candle_result.extra["data"][0].get("count", 0)
        
        # Get total count of indicators
        count_indicator_query = f"SELECT COUNT(*) as count FROM {indicators_table_name} WHERE symbol = ?"
        count_indicator_result = await database_service.execute_query(
            QueryRequest(query=count_indicator_query, parameters=(symbol.upper(),))
        )
        indicators_count = 0
        if count_indicator_result.success and count_indicator_result.extra.get("data"):
            indicators_count = count_indicator_result.extra["data"][0].get("count", 0)
        
        logger.info(f"| 📊 Database contains:")
        logger.info(f"|    - {candles_count} candle records")
        logger.info(f"|    - {indicators_count} indicator records")
        
        # Get sample data (latest 10 records)
        data_result = await candle_handler.get_data(
            symbol=symbol,
            limit=10  # Get latest 10 records as sample
        )
        
        # Step 8: Display sample data
        if candles_count > 0:
            logger.info(f"| 📋 Sample candle data (first record):")
            first_candle = data_result["candles"][0]
            logger.info(f"|    Timestamp: {first_candle.get('timestamp_local')}")
            logger.info(f"|    Open: {first_candle.get('open')}, High: {first_candle.get('high')}")
            logger.info(f"|    Low: {first_candle.get('low')}, Close: {first_candle.get('close')}")
            logger.info(f"|    Volume: {first_candle.get('volume')}")
            
            if indicators_count > 0:
                logger.info(f"| 📋 Sample indicator data (first record):")
                first_indicator = data_result["indicators"][0]
                indicator_names = [k for k in first_indicator.keys() if k not in ['id', 'timestamp', 'timestamp_utc', 'timestamp_local', 'symbol', 'created_at']]
                logger.info(f"|    Indicators: {', '.join(indicator_names[:5])}{'...' if len(indicator_names) > 5 else ''}")
        
        # Step 9: Cleanup
        await database_service.disconnect()
        logger.info(f"| ✅ Database disconnected")
        
    except Exception as e:
        logger.error(f"| ❌ Error in fetch_and_store_hyperliquid_data: {e}", exc_info=True)
        raise


async def main():
    args = parse_args()
    
    # Initialize configuration
    config.initialize(args.config, args)
    logger.initialize(config)
    logger.info(f"| Config: {config.pretty_text}")
    
    """Main function."""
    # Configuration
    symbol = "BTC"
    end_time = int(time.time() * 1000)
    start_time = end_time - 1000 * 60 * 60 * 24 * 7
    start_time = int(start_time)
    end_time = int(end_time)
    interval = "1m"  # 1 minute interval
    db_dir = "workdir/hyperliquid_data"
    
    logger.info("=" * 80)
    logger.info("Hyperliquid Data Fetching and Storage Script")
    logger.info("=" * 80)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Start Time (local): {start_time}")
    logger.info(f"End Time (local): {end_time}")
    logger.info(f"Interval: {interval}")
    logger.info(f"Database Directory: {db_dir}")
    logger.info("=" * 80)
    logger.info("")
    
    await fetch_and_store_hyperliquid_data(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        interval=interval,
        db_dir=db_dir
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Script completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
