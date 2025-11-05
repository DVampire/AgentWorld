"""Bars data handler for Alpaca streaming data."""
from typing import Optional, Union, List, Dict
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timezone

from src.logger import logger
from src.environments.database.service import DatabaseService
from src.environments.database.types import CreateTableRequest, InsertRequest, QueryRequest
from src.environments.alpacaentry.exceptions import AlpacaError


class BarsHandler:
    """Handler for bars data with streaming, caching, and technical indicators."""
    
    def __init__(self, database_service: DatabaseService):
        """Initialize bars handler.
        
        Args:
            database_service: Database service instance
        """
        self.database_service = database_service
        
        # Cache for bars data (for fast indicator calculation)
        # Key: symbol, Value: DataFrame with recent bars (max 200 rows)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_limit: int = 100  # Maximum number of bars to cache per symbol
    
    def _sanitize_table_name(self, symbol: str) -> str:
        """Sanitize symbol name to be used as table name."""
        # Replace invalid characters with underscore
        table_name = symbol.replace("/", "_").replace(".", "_").replace("-", "_")
        # Remove any other invalid characters
        table_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)
        return f"data_{table_name}"
    
    async def ensure_table_exists(self, symbol: str) -> None:
        """Ensure bars table and indicators table exist for a symbol.
        
        Args:
            symbol: Symbol name
        """
        base_name = self._sanitize_table_name(symbol)
        bars_table_name = f"{base_name}_bars"
        indicators_table_name = f"{base_name}_indicators"
        
        # Ensure bars table exists
        check_bars_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{bars_table_name}'"
        check_bars_result = await self.database_service.execute_query(
            QueryRequest(query=check_bars_query)
        )
        
        if not (check_bars_result.success and check_bars_result.extra.get("data")):
            # Create bars table
            columns = [
                {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
                {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "open", "type": "REAL"},
                {"name": "high", "type": "REAL"},
                {"name": "low", "type": "REAL"},
                {"name": "close", "type": "REAL"},
                {"name": "volume", "type": "REAL"},
                {"name": "trade_count", "type": "INTEGER"},
                {"name": "vwap", "type": "REAL"},
                {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
            ]
            
            create_request = CreateTableRequest(
                table_name=bars_table_name,
                columns=columns,
                primary_key=None  # Primary key is already in the id column constraints
            )
            result = await self.database_service.create_table(create_request)
            if not result.success:
                logger.error(f"Failed to create bars table {bars_table_name}: {result.message}")
                raise AlpacaError(f"Failed to create bars table {bars_table_name}: {result.message}")
            
            # Create index for performance optimization
            index_name = f"{bars_table_name}_timestamp_id_idx"
            index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {bars_table_name}(timestamp DESC, id DESC)"
            index_result = await self.database_service.execute_query(QueryRequest(query=index_query))
            if not index_result.success:
                logger.warning(f"Failed to create index {index_name} for {bars_table_name}: {index_result.message}")
        
        # Ensure indicators table exists
        check_indicators_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{indicators_table_name}'"
        check_indicators_result = await self.database_service.execute_query(
            QueryRequest(query=check_indicators_query)
        )
        
        if not (check_indicators_result.success and check_indicators_result.extra.get("data")):
            # Create indicators table with all technical indicators
            columns = [
                {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
                {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                # Trend indicators
                {"name": "sma_20", "type": "REAL"},
                {"name": "sma_50", "type": "REAL"},
                {"name": "ema_20", "type": "REAL"},
                {"name": "ema_50", "type": "REAL"},
                {"name": "macd", "type": "REAL"},
                {"name": "macd_signal", "type": "REAL"},
                {"name": "macd_hist", "type": "REAL"},
                {"name": "adx", "type": "REAL"},
                {"name": "sar", "type": "REAL"},
                # Momentum indicators
                {"name": "rsi", "type": "REAL"},
                {"name": "stoch_k", "type": "REAL"},
                {"name": "stoch_d", "type": "REAL"},
                {"name": "cci", "type": "REAL"},
                # Volatility indicators
                {"name": "bb_upper", "type": "REAL"},
                {"name": "bb_middle", "type": "REAL"},
                {"name": "bb_lower", "type": "REAL"},
                {"name": "atr", "type": "REAL"},
                # Volume indicators
                {"name": "obv", "type": "REAL"},
                {"name": "mfi", "type": "REAL"},
                # Structure indicators (Pivot Points and Ichimoku)
                {"name": "pivot_point", "type": "REAL"},
                {"name": "pivot_resistance1", "type": "REAL"},
                {"name": "pivot_resistance2", "type": "REAL"},
                {"name": "pivot_support1", "type": "REAL"},
                {"name": "pivot_support2", "type": "REAL"},
                {"name": "ichimoku_tenkan", "type": "REAL"},
                {"name": "ichimoku_kijun", "type": "REAL"},
                {"name": "ichimoku_senkou_a", "type": "REAL"},
                {"name": "ichimoku_senkou_b", "type": "REAL"},
                {"name": "ichimoku_chikou", "type": "REAL"},
                {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
            ]
            
            create_request = CreateTableRequest(
                table_name=indicators_table_name,
                columns=columns,
                primary_key=None
            )
            result = await self.database_service.create_table(create_request)
            if not result.success:
                logger.error(f"Failed to create indicators table {indicators_table_name}: {result.message}")
                raise AlpacaError(f"Failed to create indicators table {indicators_table_name}: {result.message}")
    
    def _normalize_timestamp(self, timestamp_value) -> str:
        """Normalize timestamp to 'YYYY-MM-DD HH:MM:SS' format string.
        
        Args:
            timestamp_value: Timestamp value (can be datetime, Timestamp object, or string)
            
        Returns:
            Formatted timestamp string
        """
        if timestamp_value is None:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        if hasattr(timestamp_value, 'strftime'):
            # datetime object or Timestamp object
            return timestamp_value.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(timestamp_value, str):
            # Already a string, assume it's in correct format
            return timestamp_value
        else:
            # Fallback: convert to string
            return str(timestamp_value)
    
    def _prepare_data_for_insert(self, data: Dict, symbol: str) -> Dict:
        """Prepare bars data dictionary for database insertion.
        
        Args:
            data: Data dictionary from Alpaca stream
            symbol: Symbol name
            
        Returns:
            Prepared data dictionary for database insertion
        """
        # Normalize timestamp
        timestamp_value = data.get("timestamp")
        timestamp_str = self._normalize_timestamp(timestamp_value)
        
        # Support both raw_data format (single letters) and object format (full names)
        db_data = {
            "timestamp": timestamp_str,
            "symbol": symbol,
            "open": data.get("open") if "open" in data else data.get("o"),
            "high": data.get("high") if "high" in data else data.get("h"),
            "low": data.get("low") if "low" in data else data.get("l"),
            "close": data.get("close") if "close" in data else data.get("c"),
            "volume": data.get("volume") if "volume" in data else data.get("v"),
            "trade_count": data.get("trade_count") if "trade_count" in data else data.get("n"),
            "vwap": data.get("vwap") if "vwap" in data else data.get("vw"),
        }
        
        return db_data
    
    async def stream_insert(self, data: Dict, symbol: str, asset_type=None) -> bool:
        """Insert bars data from stream.
        
        Args:
            data: Raw bars data dictionary from Alpaca stream
            symbol: Symbol name
            asset_type: Optional asset type (not used for bars, kept for compatibility)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure table exists
            await self.ensure_table_exists(symbol)
            
            # Prepare data for insertion
            db_data = self._prepare_data_for_insert(data, symbol)
            
            # Insert into database
            base_name = self._sanitize_table_name(symbol)
            table_name = f"{base_name}_bars"
            
            insert_request = InsertRequest(
                table_name=table_name,
                data=db_data
            )
            
            # Debug: Log what we're trying to insert
            logger.debug(f"| 🔍 Attempting to insert bars data for {symbol} into {table_name}: {db_data}")
            
            result = await self.database_service.insert_data(insert_request)
            
            if not result.success:
                logger.error(f"| ❌ Failed to insert bars data for {symbol}: {result.message}")
                logger.error(f"| ❌ Insert request: table={table_name}, data={db_data}")
                return False
            
            # Verify insertion by querying the table
            verify_query = f"SELECT COUNT(*) as count FROM {table_name}"
            verify_result = await self.database_service.execute_query(QueryRequest(query=verify_query))
            if verify_result.success:
                count = verify_result.extra.get("data", [{}])[0].get("count", 0)
                logger.debug(f"| ✅ Bars data inserted for {symbol}. Total rows in {table_name}: {count}")
            else:
                logger.warning(f"| ⚠️  Insert succeeded but couldn't verify count for {symbol}")
            
            # Update cache
            self._update_cache(symbol, db_data)
            
            # Calculate and store indicators
            await self._calculate_and_store_indicators(symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"Error inserting bars data for {symbol}: {e}")
            return False
    
    def _update_cache(self, symbol: str, new_bar: Dict) -> None:
        """Update bars cache with a new bar record.
        
        Args:
            symbol: Symbol name
            new_bar: New bar data dictionary to add to cache
        """
        try:
            # Convert new_bar to DataFrame row
            new_row = pd.DataFrame([new_bar])
            
            # Get current cache or create empty DataFrame
            if symbol in self._cache and not self._cache[symbol].empty:
                cached_df = self._cache[symbol]
                
                # Check if this bar already exists (by timestamp and id if available)
                if 'id' in new_bar and 'id' in cached_df.columns:
                    if new_bar['id'] in cached_df['id'].values:
                        # Bar already in cache, skip
                        return
                
                # Append new bar to cache
                cached_df = pd.concat([cached_df, new_row], ignore_index=True)
                
                # Sort by timestamp and id to maintain order
                if 'timestamp' in cached_df.columns:
                    cached_df = cached_df.sort_values('timestamp', ascending=True)
                    if 'id' in cached_df.columns:
                        cached_df = cached_df.sort_values(['timestamp', 'id'], ascending=True)
                
                # Remove oldest bars if cache exceeds limit
                if len(cached_df) > self._cache_limit:
                    cached_df = cached_df.tail(self._cache_limit).reset_index(drop=True)
                
                self._cache[symbol] = cached_df
            else:
                # Cache is empty, just set the new bar
                self._cache[symbol] = new_row
                
        except Exception as e:
            logger.warning(f"Failed to update bars cache for {symbol}: {e}")
            # On error, invalidate cache to force reload from database next time
            if symbol in self._cache:
                del self._cache[symbol]
    
    async def get_recent_bars(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        """Get recent bars data for calculating technical indicators.
        
        Uses cache for fast retrieval. If cache exists, returns cached data (up to limit).
        If cache is empty or missing, loads from database and populates cache.
        
        Args:
            symbol: Symbol name
            limit: Maximum number of bars to retrieve (default 200 for most indicators)
            
        Returns:
            pandas DataFrame with bars records ordered by timestamp
        """
        # Check cache first
        if symbol in self._cache and not self._cache[symbol].empty:
            cached_df = self._cache[symbol]
            # Return cached data up to limit
            if len(cached_df) <= limit:
                return cached_df.copy()
            else:
                # Return last 'limit' rows from cache
                return cached_df.tail(limit).copy()
        
        # Cache miss: load from database
        base_name = self._sanitize_table_name(symbol)
        table_name = f"{base_name}_bars"
        
        # Get recent bars ordered by timestamp
        query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC, id DESC LIMIT {limit}"
        result = await self.database_service.execute_query(QueryRequest(query=query))
        
        if not result.success:
            logger.warning(f"Failed to get recent bars for {symbol}: {result.message}")
            return pd.DataFrame()
        
        bars = result.extra.get("data", [])
        if not bars:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(bars)
        
        # Reverse to get chronological order
        df = df.iloc[::-1].reset_index(drop=True)
        
        # Update cache
        self._cache[symbol] = df.copy()
        
        return df
    
    def calculate_indicators(self, bars: pd.DataFrame) -> Optional[Dict]:
        """Calculate technical indicators from bars data.
        
        Args:
            bars: DataFrame with fields: open, high, low, close, volume, timestamp
            
        Returns:
            Dictionary of calculated indicators, or None if insufficient data
        """
        if bars.empty or len(bars) < 2:
            return None
        
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in bars.columns]
        if missing_columns:
            logger.warning(f"Missing required columns for indicator calculation: {missing_columns}")
            return None
        
        # Convert to numpy arrays (TA-Lib can accept pandas Series directly, but we ensure float64)
        try:
            closes = bars['close'].astype(float).values
            highs = bars['high'].astype(float).values
            lows = bars['low'].astype(float).values
            opens = bars['open'].astype(float).values
            volumes = bars['volume'].fillna(0.0).astype(float).values
            
            if len(closes) == 0 or len(closes) != len(bars):
                return None
            
            indicators = {}
            
            # Trend indicators
            if len(closes) >= 20:
                indicators["sma_20"] = float(talib.SMA(closes, timeperiod=20)[-1])
                indicators["ema_20"] = float(talib.EMA(closes, timeperiod=20)[-1])
            if len(closes) >= 50:
                indicators["sma_50"] = float(talib.SMA(closes, timeperiod=50)[-1])
                indicators["ema_50"] = float(talib.EMA(closes, timeperiod=50)[-1])
            
            # MACD (12, 26, 9)
            if len(closes) >= 26:
                macd, macd_signal, macd_hist = talib.MACD(closes, fastperiod=12, slowperiod=26, signalperiod=9)
                indicators["macd"] = float(macd[-1]) if not np.isnan(macd[-1]) else None
                indicators["macd_signal"] = float(macd_signal[-1]) if not np.isnan(macd_signal[-1]) else None
                indicators["macd_hist"] = float(macd_hist[-1]) if not np.isnan(macd_hist[-1]) else None
            
            # ADX (needs at least 14 periods, but usually needs more for accuracy)
            if len(highs) >= 14:
                adx = talib.ADX(highs, lows, closes, timeperiod=14)
                indicators["adx"] = float(adx[-1]) if not np.isnan(adx[-1]) else None
            
            # SAR
            if len(highs) >= 2:
                sar = talib.SAR(highs, lows, acceleration=0.02, maximum=0.2)
                indicators["sar"] = float(sar[-1]) if not np.isnan(sar[-1]) else None
            
            # Momentum indicators
            # RSI
            if len(closes) >= 14:
                rsi = talib.RSI(closes, timeperiod=14)
                indicators["rsi"] = float(rsi[-1]) if not np.isnan(rsi[-1]) else None
            
            # Stochastic (KDJ)
            if len(highs) >= 14:
                stoch_k, stoch_d = talib.STOCH(highs, lows, closes, 
                                               fastk_period=14, 
                                               slowk_period=3, 
                                               slowd_period=3)
                indicators["stoch_k"] = float(stoch_k[-1]) if not np.isnan(stoch_k[-1]) else None
                indicators["stoch_d"] = float(stoch_d[-1]) if not np.isnan(stoch_d[-1]) else None
            
            # CCI
            if len(highs) >= 14:
                cci = talib.CCI(highs, lows, closes, timeperiod=14)
                indicators["cci"] = float(cci[-1]) if not np.isnan(cci[-1]) else None
            
            # Volatility indicators
            # Bollinger Bands
            if len(closes) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
                indicators["bb_upper"] = float(bb_upper[-1]) if not np.isnan(bb_upper[-1]) else None
                indicators["bb_middle"] = float(bb_middle[-1]) if not np.isnan(bb_middle[-1]) else None
                indicators["bb_lower"] = float(bb_lower[-1]) if not np.isnan(bb_lower[-1]) else None
            
            # ATR
            if len(highs) >= 14:
                atr = talib.ATR(highs, lows, closes, timeperiod=14)
                indicators["atr"] = float(atr[-1]) if not np.isnan(atr[-1]) else None
            
            # Volume indicators
            # OBV
            if len(closes) >= 2 and len(volumes) >= 2:
                obv = talib.OBV(closes, volumes)
                indicators["obv"] = float(obv[-1]) if not np.isnan(obv[-1]) else None
            
            # MFI
            if len(highs) >= 14:
                mfi = talib.MFI(highs, lows, closes, volumes, timeperiod=14)
                indicators["mfi"] = float(mfi[-1]) if not np.isnan(mfi[-1]) else None
            
            # Structure indicators
            # Pivot Points (calculated from current bar's high, low, close)
            if len(bars) >= 1:
                current_bar = bars.iloc[-1]
                high = float(current_bar['high'])
                low = float(current_bar['low'])
                close = float(current_bar['close'])
                if high > 0 and low > 0 and close > 0:
                    pivot = (high + low + close) / 3
                    indicators["pivot_point"] = float(pivot)
                    indicators["pivot_resistance1"] = float(2 * pivot - low)
                    indicators["pivot_resistance2"] = float(pivot + (high - low))
                    indicators["pivot_support1"] = float(2 * pivot - high)
                    indicators["pivot_support2"] = float(pivot - (high - low))
            
            # Ichimoku (needs 52 periods for senkou_b)
            if len(highs) >= 52:
                # Tenkan-sen (conversion line): (highest high + lowest low) / 2 for 9 periods
                tenkan_high = talib.MAX(highs, timeperiod=9)
                tenkan_low = talib.MIN(lows, timeperiod=9)
                tenkan = (tenkan_high + tenkan_low) / 2
                
                # Kijun-sen (base line): (highest high + lowest low) / 2 for 26 periods
                kijun_high = talib.MAX(highs, timeperiod=26)
                kijun_low = talib.MIN(lows, timeperiod=26)
                kijun = (kijun_high + kijun_low) / 2
                
                # Senkou Span A: (Tenkan + Kijun) / 2
                senkou_a = (tenkan + kijun) / 2
                
                # Senkou Span B: (highest high + lowest low) / 2 for 52 periods
                senkou_b_high = talib.MAX(highs, timeperiod=52)
                senkou_b_low = talib.MIN(lows, timeperiod=52)
                senkou_b = (senkou_b_high + senkou_b_low) / 2
                
                indicators["ichimoku_tenkan"] = float(tenkan[-1]) if not np.isnan(tenkan[-1]) else None
                indicators["ichimoku_kijun"] = float(kijun[-1]) if not np.isnan(kijun[-1]) else None
                indicators["ichimoku_senkou_a"] = float(senkou_a[-1]) if not np.isnan(senkou_a[-1]) else None
                indicators["ichimoku_senkou_b"] = float(senkou_b[-1]) if not np.isnan(senkou_b[-1]) else None
                # Chikou is plotted 26 periods back, so we use the close from 26 periods ago
                if len(closes) >= 26:
                    indicators["ichimoku_chikou"] = float(closes[-26]) if not np.isnan(closes[-26]) else None
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return None
    
    async def _calculate_and_store_indicators(self, symbol: str) -> None:
        """Calculate and store technical indicators for the bars.
        
        Args:
            symbol: Symbol name
        """
        try:
            # Ensure indicators table exists (ensure_table_exists creates both tables)
            await self.ensure_table_exists(symbol)
            
            # Get bars (need enough for indicators like EMA50, Ichimoku)
            bars = await self.get_recent_bars(symbol, limit=200)
            if bars.empty:
                return
            
            # Calculate indicators
            indicators = self.calculate_indicators(bars)
            if not indicators:
                return
            
            # Get the latest bar's timestamp
            if bars.empty:
                return
            latest_bar = bars.iloc[-1].to_dict()
            timestamp = latest_bar.get("timestamp")
            if not timestamp:
                return
            
            # Prepare indicator data for insertion
            base_name = self._sanitize_table_name(symbol)
            table_name = f"{base_name}_indicators"
            
            indicator_data = {
                "timestamp": timestamp,
                "symbol": symbol,
            }
            
            # Add all calculated indicators
            indicator_data.update(indicators)
            
            # Check if indicator record already exists for this timestamp
            check_query = f"SELECT id FROM {table_name} WHERE timestamp = ? AND symbol = ?"
            check_result = await self.database_service.execute_query(
                QueryRequest(query=check_query, parameters=(timestamp, symbol))
            )
            
            if check_result.success and check_result.extra.get("data"):
                # Update existing record
                record_id = check_result.extra["data"][0]["id"]
                update_fields = ", ".join([f"{k} = ?" for k in indicator_data.keys() if k != "timestamp" and k != "symbol"])
                update_values = [v for k, v in indicator_data.items() if k != "timestamp" and k != "symbol"]
                update_values.append(record_id)
                
                update_query = f"UPDATE {table_name} SET {update_fields} WHERE id = ?"
                update_result = await self.database_service.execute_query(
                    QueryRequest(query=update_query, parameters=tuple(update_values))
                )
                if not update_result.success:
                    logger.error(f"Failed to update indicators for {symbol} at {timestamp}: {update_result.message}")
            else:
                # Insert new record
                insert_request = InsertRequest(
                    table_name=table_name,
                    data=indicator_data
                )
                result = await self.database_service.insert_data(insert_request)
                if not result.success:
                    logger.error(f"Failed to insert indicators for {symbol}: {result.message}")
                    
        except Exception as e:
            logger.error(f"Error calculating and storing indicators for {symbol}: {e}")
    
    async def get_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get bars data from database with corresponding technical indicators.
        
        Args:
            symbol: Symbol name
            start_date: Optional start date in format 'YYYY-MM-DD HH:MM:SS'
            end_date: Optional end date in format 'YYYY-MM-DD HH:MM:SS'
            limit: Optional limit
            
        Returns:
            List of bars records, each with an 'indicators' field containing non-NULL indicator values
        """
        base_name = self._sanitize_table_name(symbol)
        table_name = f"{base_name}_bars"
        indicators_table_name = f"{base_name}_indicators"
        
        # Check if bars table exists
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        check_result = await self.database_service.execute_query(
            QueryRequest(query=check_query)
        )
        
        if not check_result.success or not check_result.extra.get("data"):
            logger.warning(f"| ⚠️  Bars table {table_name} does not exist for {symbol}")
            return []
        
        # Debug: Check total row count
        count_query = f"SELECT COUNT(*) as count FROM {table_name}"
        count_result = await self.database_service.execute_query(QueryRequest(query=count_query))
        if count_result.success:
            count = count_result.extra.get("data", [{}])[0].get("count", 0)
            logger.debug(f"| 🔍 Querying bars for {symbol}: table {table_name} has {count} rows")
        
        # Check if indicators table exists
        check_indicators_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{indicators_table_name}'"
        check_indicators_result = await self.database_service.execute_query(
            QueryRequest(query=check_indicators_query)
        )
        has_indicators_table = check_indicators_result.success and check_indicators_result.extra.get("data")
        
        # Build query for bars based on whether date range is provided
        if start_date and end_date:
            query = f"SELECT * FROM {table_name} WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
            parameters = (start_date, end_date)
            if limit:
                query += f" LIMIT {limit}"
        else:
            if limit:
                query = f"SELECT * FROM {table_name} ORDER BY timestamp DESC, id DESC LIMIT {limit}"
                parameters = None
            else:
                # Get latest timestamp
                max_timestamp_query = f"SELECT MAX(timestamp) as max_ts FROM {table_name}"
                max_ts_result = await self.database_service.execute_query(
                    QueryRequest(query=max_timestamp_query)
                )
                
                if not max_ts_result.success or not max_ts_result.extra.get("data"):
                    return []
                
                max_ts_data = max_ts_result.extra.get("data", [])
                if not max_ts_data or not max_ts_data[0].get("max_ts"):
                    return []
                
                latest_timestamp = max_ts_data[0]["max_ts"]
                query = f"SELECT * FROM {table_name} WHERE timestamp = ? ORDER BY timestamp ASC, id ASC"
                parameters = (latest_timestamp,)
        
        # Debug: Log the query being executed
        logger.debug(f"| 🔍 Executing query for {symbol}: {query}")
        if parameters:
            logger.debug(f"| 🔍 Query parameters: {parameters}")
        
        result = await self.database_service.execute_query(
            QueryRequest(query=query, parameters=parameters)
        )
        
        if not result.success:
            logger.warning(f"| ⚠️  Failed to query bars from {table_name}: {result.message}")
            return []
        
        bars_data = result.extra.get("data", [])
        logger.debug(f"| 🔍 Query returned {len(bars_data)} rows for {symbol}")
        
        if bars_data:
            logger.debug(f"| 🔍 First row sample: {bars_data[0]}")
        else:
            logger.warning(f"| ⚠️  No bars data returned for {symbol} from table {table_name}")
        
        # If limit was specified and we're not using date range, reverse to get chronological order
        if limit and not start_date and not end_date:
            bars_data.reverse()
        
        # If indicators table exists, fetch indicators for the same timestamps
        if has_indicators_table and bars_data:
            # Get all timestamps from bars data
            timestamps = [bar.get("timestamp") for bar in bars_data if bar.get("timestamp")]
            
            indicators_dict = {}
            if timestamps:
                # Build query for indicators
                if start_date and end_date:
                    indicators_query = f"SELECT * FROM {indicators_table_name} WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
                    indicators_parameters = (start_date, end_date)
                    if limit:
                        indicators_query += f" LIMIT {limit}"
                else:
                    if limit:
                        indicators_query = f"SELECT * FROM {indicators_table_name} ORDER BY timestamp DESC, id DESC LIMIT {limit}"
                        indicators_parameters = None
                    else:
                        # Get latest timestamp
                        latest_timestamp = timestamps[-1] if timestamps else None
                        if latest_timestamp:
                            indicators_query = f"SELECT * FROM {indicators_table_name} WHERE timestamp = ? ORDER BY timestamp ASC, id ASC"
                            indicators_parameters = (latest_timestamp,)
                        else:
                            indicators_parameters = None
                            indicators_query = None
                
                # Fetch indicators
                if indicators_query:
                    indicators_result = await self.database_service.execute_query(
                        QueryRequest(query=indicators_query, parameters=indicators_parameters)
                    )
                    
                    if indicators_result.success:
                        indicators_data = indicators_result.extra.get("data", [])
                        
                        # Process indicators: group by timestamp
                        for indicator_record in indicators_data:
                            timestamp = indicator_record.get("timestamp")
                            if timestamp:
                                if timestamp not in indicators_dict:
                                    indicators_dict[timestamp] = {}
                                
                                # Add only non-NULL indicator values
                                indicator_fields = [
                                    "sma_20", 
                                    "sma_50", 
                                    "ema_20",
                                    "ema_50",
                                    "macd", 
                                    "macd_signal", 
                                    "macd_hist",
                                    "adx",
                                    "sar",
                                    "rsi", 
                                    "stoch_k",
                                    "stoch_d",
                                    "cci",
                                    "bb_upper",
                                    "bb_middle",
                                    "bb_lower",
                                    "atr",
                                    "obv", 
                                    "mfi",
                                    "pivot_point",
                                    "pivot_resistance1",
                                    "pivot_resistance2",
                                    "pivot_support1",
                                    "pivot_support2",
                                    "ichimoku_tenkan", 
                                    "ichimoku_kijun", 
                                    "ichimoku_senkou_a",
                                    "ichimoku_senkou_b", 
                                    "ichimoku_chikou"
                                ]
                                
                                for field in indicator_fields:
                                    value = indicator_record.get(field)
                                    if value is not None:
                                        indicators_dict[timestamp][field] = value
            
            # Merge indicators into bars data
            for bar in bars_data:
                timestamp = bar.get("timestamp")
                if timestamp and timestamp in indicators_dict:
                    bar["indicators"] = indicators_dict[timestamp]
                else:
                    bar["indicators"] = {}
        else:
            # No indicators table, add empty indicators dict
            for bar in bars_data:
                bar["indicators"] = {}
        
        return bars_data
    

