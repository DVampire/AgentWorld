"""Candle data handler for Hyperliquid streaming data."""
import pandas as pd
import asyncio
import time
from typing import Optional, List, Dict, Union

from src.logger import logger
from src.config import config
from src.environments.database.service import DatabaseService
from src.environments.database.types import CreateTableRequest, InsertRequest, QueryRequest, SelectRequest
from src.environments.hyperliquidentry.exceptions import HyperliquidError
from src.registry import INDICATOR
from src.utils import get_standard_timestamp

class CandleHandler:
    """Handler for candle (OHLCV) data with streaming, caching, and technical indicators."""
    
    def __init__(self, database_service: DatabaseService):
        """Initialize candle handler.
        
        Args:
            database_service: Database service instance
        """
        self.database_service = database_service
        
        # Cache for candle data (for fast indicator calculation)
        # Key: symbol, Value: DataFrame with recent candles (max 200 rows)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_limit: int = 200  # Maximum number of candles to cache per symbol
        
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
        """Preprocess data for insertion.
        
        Args:
            data: Data dictionary from Hyperliquid WebSocket stream
            {
                T: int, # close time (ms)
                c: float string, # close price
                h: float string, # high price
                i: str, # interval
                l: float string, # low price
                n: int, # trade count
                o: float string, # open price
                s: string, # symbol
                t: int, # open time (ms)
                v: float string, # volume
            }
        """
        
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
        
        timestamp_dict = get_standard_timestamp(close_time + 1000) # add 1 second to close_time to get the next minute start
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
        """Calculate indicators for a given dataframe.
        
        Calculates indicators in batches to control concurrency.
        
        Args:
            df: DataFrame with candle data
            symbol: Symbol name
            
        Returns:
            DataFrame with calculated indicators
        """
        all_indicators_df = pd.DataFrame(index=df.index)
        
        # Convert indicators dict to list of (name, obj) tuples for easier batching
        indicators_list = [(indicator_obj.indicators_name, indicator_obj) 
                          for _, indicator_obj in self._indicators_functions.items()]
        
        # Process indicators in batches
        for i in range(0, len(indicators_list), self._concurrent_coroutines):
            batch = indicators_list[i:i+self._concurrent_coroutines]
            
            indicator_tasks = []
            indicator_names = []
            for indicator_name, indicator_obj in batch:
                indicator_tasks.append(indicator_obj(df.copy()))
                indicator_names.append(indicator_name)
            
            # Wait for current batch to complete
            indicator_results = await asyncio.gather(*indicator_tasks, return_exceptions=True)
            
            # Merge results from current batch
            for indicator_name, result in zip(indicator_names, indicator_results):
                if isinstance(result, Exception):
                    logger.warning(f"| ⚠️  Error calculating indicator {indicator_name} for {symbol}: {result}")
                    continue
                if isinstance(result, pd.DataFrame) and not result.empty:
                    # Merge indicator columns, aligning by index
                    for col in result.columns:
                        all_indicators_df[col] = result[col]
        
        return all_indicators_df
    
    async def full_insert(self, data: List[Dict], symbol: str) -> Dict:
        """Insert candle data from list of dictionaries (full update for initial 60 minutes).
        
        This method:
        1. Preprocesses all data
        2. Inserts all data into the database
        3. Updates cache
        4. Calculates indicators in parallel using self._indicators_functions
        5. Inserts indicators into indicators database
        
        Args:
            data: List of candle data dictionaries
            symbol: Symbol name (will be normalized to uppercase)
            
        Returns:
            Dict with success status and message
        """
        try:
            # Normalize symbol to uppercase
            symbol = symbol.upper() if symbol else ""
            
            if not data:
                logger.warning(f"| ⚠️  No data provided for full_insert for {symbol}")
                return {"success": False, "message": "No data provided"}
            
            # Ensure table exists
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
            
            # Step 2: Insert all data into database using insert_data method
            logger.info(f"| 💾 Inserting {len(processed_data)} records into {candle_table_name}")
            
            if not processed_data:
                logger.warning(f"| ⚠️  No processed data to insert for {symbol}")
                return {"success": False, "message": "No processed data to insert"}
            
            insert_request = InsertRequest(
                table_name=candle_table_name,
                data=processed_data,
                on_conflict="REPLACE"  # Use INSERT OR REPLACE to handle duplicate (symbol, timestamp)
            )
            result = await self.database_service.insert_data(insert_request)
            
            if not result.success:
                logger.error(f"| ❌ Failed to insert candle data for {symbol}: {result.message}")
                return {"success": False, "message": f"Failed to insert candle data: {result.message}"}
            
            rowcount = result.extra.get("row_count", len(processed_data)) if result.extra else len(processed_data)
            logger.info(f"| ✅ Inserted {rowcount} candle records for {symbol}")
            
            # Step 3: Update cache
            logger.info(f"| 📦 Updating cache for {symbol}")
            await self._update_cache(symbol, processed_data)
            logger.info(f"| ✅ Cache updated for {symbol} (now has {len(self._cache[symbol])} records)")
            
            # Step 4: Calculate indicators in parallel (with concurrency control)
            logger.info(f"| 📊 Calculating indicators for {symbol} using {len(self._indicators_functions)} indicator(s)")
            
            # Prepare DataFrame for indicators
            df = self._cache[symbol].copy()
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            
            # Ensure we have numeric columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Calculate indicators using the dedicated method (with concurrency control)
            all_indicators_df = await self._calculate_indicators(df, symbol)
            
            if all_indicators_df.empty or len(all_indicators_df.columns) == 0:
                logger.warning(f"| ⚠️  No indicators calculated for {symbol}")
                return {"success": True, "message": f"Inserted {len(processed_data)} records, but no indicators calculated"}
            
            # Step 5: Insert indicators into database
            # For each row in the cache, create an indicator record
            logger.info(f"| 💾 Inserting indicators for {len(df)} records into {indicators_table_name}")
            
            indicators_to_insert = []
            for idx, row in df.iterrows():
                # Get corresponding indicator values
                # Build indicator_data in the same order as table definition:
                # timestamp, timestamp_utc, timestamp_local, symbol, then all indicators in self._indicators_name order
                indicator_data = {
                    "timestamp": int(row["timestamp"]),
                    "timestamp_utc": row.get("timestamp_utc", ""),
                    "timestamp_local": row.get("timestamp_local", ""),
                    "symbol": symbol,
                }
                
                # Add indicator values from all_indicators_df in the order defined by self._indicators_name
                if idx in all_indicators_df.index:
                    indicator_row = all_indicators_df.loc[idx]
                    # Use self._indicators_name order to ensure consistency with table definition
                    for indicator_name in self._indicators_name:
                        if indicator_name in indicator_row.index:
                            value = indicator_row[indicator_name]
                            if pd.notna(value):
                                indicator_data[indicator_name] = float(value)
                            else:
                                indicator_data[indicator_name] = None
                
                indicators_to_insert.append(indicator_data)
            
            # Batch insert indicators using insert_data method
            if indicators_to_insert:
                insert_request = InsertRequest(
                    table_name=indicators_table_name,
                    data=indicators_to_insert,
                    on_conflict="REPLACE"  # Use INSERT OR REPLACE to handle duplicate (symbol, timestamp)
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
       
    async def stream_insert(self, data: Dict, symbol: str) -> Dict:
        """Insert candle data from stream (single row).
        
        This method:
        1. Preprocesses the data
        2. Inserts/Replaces the data into the database
        3. Updates cache
        4. Calculates indicators using the full cache DataFrame
        5. Inserts only the latest indicator record into database
        
        Args:
            data: Candle data from Hyperliquid WebSocket stream (processed format)
            symbol: Symbol name (can be lowercase or uppercase)
            
        Returns:
            Insert result
        """
        try:
            # Normalize symbol to uppercase for consistency with database
            symbol = symbol.upper() if symbol else data.get("symbol", "").upper()
            
            if not data:
                logger.warning(f"| ⚠️  No data provided for stream_insert for {symbol}")
                return {"success": False, "message": "No data provided"}
            
            # Ensure table exists
            await self.ensure_table_exists(symbol)
            
            base_name = self._sanitize_table_name(symbol)
            candle_table_name = f"{base_name}_candle"
            indicators_table_name = f"{base_name}_indicators"
            
            # Step 1: Preprocess data
            processed_data = await self._preprocess_data(data, symbol)
            
            # Step 2: Insert single row into database
            logger.info(f"| 💾 Inserting candle for {symbol}: timestamp={processed_data.get('timestamp')}")
            
            insert_request = InsertRequest(
                table_name=candle_table_name,
                data=processed_data,
                on_conflict="REPLACE"  # Use INSERT OR REPLACE to handle duplicate (symbol, timestamp)
            )
            result = await self.database_service.insert_data(insert_request)
            
            if not result.success:
                logger.error(f"| ❌ Failed to insert candle for {symbol}: {result.message}")
                return {"success": False, "message": f"Failed to insert candle data: {result.message}"}
            
            logger.info(f"| ✅ Inserted candle for {symbol} (timestamp: {processed_data.get('timestamp')})")
            
            # Step 3: Update cache
            await self._update_cache(symbol, processed_data)
            
            # Step 4: Calculate indicators using full cache DataFrame
            logger.info(f"| 📊 Calculating indicators for {symbol} using {len(self._indicators_functions)} indicator(s)")
            
            # Prepare DataFrame for indicators
            df = self._cache[symbol].copy()
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            
            # Ensure we have numeric columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Calculate indicators using the dedicated method (with concurrency control)
            all_indicators_df = await self._calculate_indicators(df, symbol)
            
            if all_indicators_df.empty or len(all_indicators_df.columns) == 0:
                logger.debug(f"| ⚠️  No indicators calculated for {symbol}")
                return {"success": True, "message": "Candle data inserted, but no indicators calculated"}
            
            # Step 5: Insert only the latest indicator record
            # Get the last row (most recent)
            latest_row = df.iloc[-1]
            latest_timestamp = int(latest_row["timestamp"])
            
            # Get corresponding indicator values for the latest row
            latest_idx = df.index[-1]
            # Build indicator_data in the same order as table definition:
            # timestamp, timestamp_utc, timestamp_local, symbol, then all indicators in self._indicators_name order
            indicator_data = {
                "timestamp": latest_timestamp,
                "timestamp_utc": latest_row.get("timestamp_utc", ""),
                "timestamp_local": latest_row.get("timestamp_local", ""),
                "symbol": symbol,
            }
            
            if latest_idx in all_indicators_df.index:
                indicator_row = all_indicators_df.loc[latest_idx]
                # Use self._indicators_name order to ensure consistency with table definition
                for indicator_name in self._indicators_name:
                    if indicator_name in indicator_row.index:
                        value = indicator_row[indicator_name]
                        if pd.notna(value):
                            indicator_data[indicator_name] = float(value)
                        else:
                            indicator_data[indicator_name] = None
            
            # Insert single indicator record using insert_data
            if indicator_data:
                insert_request = InsertRequest(
                    table_name=indicators_table_name,
                    data=indicator_data,
                    on_conflict="REPLACE"  # Use INSERT OR REPLACE to handle duplicate (symbol, timestamp)
                )
                result = await self.database_service.insert_data(insert_request)
                
                if result.success:
                    logger.info(f"| ✅ Inserted indicator for {symbol} at timestamp {latest_timestamp}")
                else:
                    logger.warning(f"| ⚠️  Failed to insert indicator for {symbol}: {result.message}")
            
            return {"success": True, "message": "Candle data and latest indicator inserted"}
            
        except Exception as e:
            logger.error(f"| ❌ Exception in stream_insert for {symbol}: {e}", exc_info=True)
            return {"success": False, "message": str(e)}
    
    async def _update_cache(self, symbol: str, candle_data: Union[Dict, List[Dict]]) -> None:
        """Update cache with new candle data.
        
        Supports both single row (Dict) and multiple rows (List[Dict]).
        
        Args:
            symbol: Symbol name
            candle_data: Candle data dictionary or list of dictionaries
        """
        if symbol not in self._cache:
            self._cache[symbol] = pd.DataFrame()
        
        # Handle both single dict and list of dicts
        if isinstance(candle_data, dict):
            new_data = pd.DataFrame([candle_data])
        else:
            # List of dicts
            new_data = pd.DataFrame(candle_data)
        
        df = self._cache[symbol]
        
        # Merge with existing cache, remove duplicates, and keep latest
        if df.empty:
            df = new_data.copy()
        else:
            df_combined = pd.concat([df, new_data], ignore_index=True)
            # Remove duplicates based on timestamp, keeping the last occurrence
            # Note: symbol is already filtered by the cache key, so we only need timestamp
            df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
            # Sort by timestamp
            if "timestamp" in df_combined.columns:
                df_combined = df_combined.sort_values("timestamp")
            df = df_combined
        
        # Limit cache size
        if len(df) > self._cache_limit:
            df = df.tail(self._cache_limit)
        
        self._cache[symbol] = df
    
    async def get_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, List[Dict]]:
        """Get candle data and indicators from database.
        
        Args:
            symbol: Symbol name
            start_date: Optional start date in format 'YYYY-MM-DD HH:MM:SS'
            end_date: Optional end date in format 'YYYY-MM-DD HH:MM:SS'
            limit: Optional limit
            
        Returns:
            Dict with 'candles' and 'indicators' fields, each containing a list of records
        """
        # Normalize symbol to uppercase for consistency with database
        symbol = symbol.upper() if symbol else ""
        base_name = self._sanitize_table_name(symbol)
        candle_table_name = f"{base_name}_candle"
        indicators_table_name = f"{base_name}_indicators"
        
        # Build select request for candles
        if start_date and end_date:
            # Date range query
            candle_request = SelectRequest(
                table_name=candle_table_name,
                where_clause="symbol = :symbol AND timestamp >= :start_date AND timestamp <= :end_date",
                where_params={"symbol": symbol, "start_date": start_date, "end_date": end_date},
                order_by="timestamp ASC",
                limit=limit
            )
        else:
            if limit:
                # Limit query - get latest N records
                candle_request = SelectRequest(
                    table_name=candle_table_name,
                    where_clause="symbol = :symbol",
                    where_params={"symbol": symbol},
                    order_by="timestamp DESC",
                    limit=limit
                )
            else:
                # Get latest timestamp first
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
        
        # Query candles using select_data
        candle_result = await self.database_service.select_data(candle_request)
        
        if not candle_result.success:
            logger.warning(f"| ⚠️  Failed to query candles: {candle_result.message}")
            return {"candles": [], "indicators": []}
        
        candle_data = candle_result.extra.get("data", [])
        
        # If limit was specified and we're not using date range, reverse to get chronological order
        if limit and not start_date and not end_date:
            candle_data.reverse()
        
        # Build select request for indicators (same conditions as candles)
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
                # Use same latest timestamp as candles
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
        
        # Query indicators using select_data
        indicators_result = await self.database_service.select_data(indicators_request)
        
        if indicators_result.success:
            indicators_data = indicators_result.extra.get("data", [])
            
            # If limit was specified and we're not using date range, reverse to get chronological order
            if limit and not start_date and not end_date:
                indicators_data.reverse()
        else:
            indicators_data = []
        
        return {
            "candles": candle_data,
            "indicators": indicators_data
        }

