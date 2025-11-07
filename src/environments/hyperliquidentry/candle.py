"""Candle data handler for Hyperliquid streaming data."""
from typing import Optional, List, Dict
import pandas as pd
import talib
from datetime import datetime, timezone

from src.logger import logger
from src.environments.database.service import DatabaseService
from src.environments.database.types import CreateTableRequest, InsertRequest, QueryRequest
from src.environments.hyperliquidentry.exceptions import HyperliquidError


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
            symbol: Symbol name
        """
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
                {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "open_time", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "close_time", "type": "TEXT"},
                {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "interval", "type": "TEXT"},  # e.g., "1m", "5m", "1h", "1d"
                {"name": "open", "type": "REAL"},
                {"name": "high", "type": "REAL"},
                {"name": "low", "type": "REAL"},
                {"name": "close", "type": "REAL"},
                {"name": "volume", "type": "REAL"},
                {"name": "quote_volume", "type": "REAL"},  # Quote asset volume
                {"name": "trade_count", "type": "INTEGER"},
                {"name": "taker_buy_base_volume", "type": "REAL"},
                {"name": "taker_buy_quote_volume", "type": "REAL"},
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
            
            # Create index for performance optimization
            index_name = f"{candle_table_name}_timestamp_idx"
            index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {candle_table_name}(timestamp DESC, open_time DESC)"
            index_result = await self.database_service.execute_query(QueryRequest(query=index_query))
            if not index_result.success:
                logger.warning(f"Failed to create index {index_name} for {candle_table_name}: {index_result.message}")
        
        # Ensure indicators table exists (same as binance klines)
        check_indicators_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{indicators_table_name}'"
        check_indicators_result = await self.database_service.execute_query(
            QueryRequest(query=check_indicators_query)
        )
        
        if not (check_indicators_result.success and check_indicators_result.extra.get("data")):
            # Create indicators table with all technical indicators
            columns = [
                {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
                {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "open_time", "type": "TEXT", "constraints": "NOT NULL"},
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
                # Structure indicators
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
                raise HyperliquidError(f"Failed to create indicators table {indicators_table_name}: {result.message}")
            
            # Create index for performance optimization
            index_name = f"{indicators_table_name}_timestamp_idx"
            index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {indicators_table_name}(timestamp DESC, open_time DESC)"
            index_result = await self.database_service.execute_query(QueryRequest(query=index_query))
            if not index_result.success:
                logger.warning(f"Failed to create index {index_name} for {indicators_table_name}: {index_result.message}")
    
    def _normalize_timestamp(self, timestamp_value) -> str:
        """Normalize timestamp to 'YYYY-MM-DD HH:MM:SS' format string in UTC.
        
        Args:
            timestamp_value: Timestamp (can be int, float, or datetime)
            
        Returns:
            'YYYY-MM-DD HH:MM:SS' format timestamp string in UTC
        """
        if isinstance(timestamp_value, (int, float)):
            # Hyperliquid uses milliseconds
            if timestamp_value > 1e10:
                # Already in milliseconds
                dt = datetime.fromtimestamp(timestamp_value / 1000, tz=timezone.utc)
            else:
                # In seconds
                dt = datetime.fromtimestamp(timestamp_value, tz=timezone.utc)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(timestamp_value, datetime):
            if timestamp_value.tzinfo is None:
                timestamp_value = timestamp_value.replace(tzinfo=timezone.utc)
            else:
                timestamp_value = timestamp_value.astimezone(timezone.utc)
            return timestamp_value.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(timestamp_value, str):
            # If already in 'YYYY-MM-DD HH:MM:SS' format, return as is
            if len(timestamp_value) == 19 and timestamp_value[10] == ' ':
                return timestamp_value
            # Try to parse and convert to UTC
            try:
                dt = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                return timestamp_value
        else:
            return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
    async def stream_insert(self, data: Dict, symbol: str) -> Dict:
        """Insert candle data from stream.
        
        Args:
            data: Candle data from Hyperliquid WebSocket stream (processed format)
            symbol: Symbol name
            
        Returns:
            Insert result
        """
        await self.ensure_table_exists(symbol)
        
        base_name = self._sanitize_table_name(symbol)
        candle_table_name = f"{base_name}_candle"
        
        logger.info(f"| 🔍 stream_insert called for {symbol}, data keys: {list(data.keys()) if isinstance(data, dict) else 'not a dict'}")
        
        # Processed data format from WebSocket class:
        # {
        #   "symbol": "BTC",
        #   "interval": "1m",
        #   "timestamp": "YYYY-MM-DD HH:MM:SS" (minute start time, UTC),
        #   "open_time": "YYYY-MM-DD HH:MM:SS" (minute start time, UTC),
        #   "close_time": "YYYY-MM-DD HH:MM:SS" (minute end time, UTC),
        #   "open": float, "high": float, "low": float, "close": float,
        #   "volume": float, "quote_volume": float, "trade_count": int,
        #   "taker_buy_base_volume": float, "taker_buy_quote_volume": float,
        #   "is_closed": True
        # }
        
        if not isinstance(data, dict):
            logger.error(f"| ❌ stream_insert: data is not a dict for {symbol}, type: {type(data)}")
            return {"success": False, "message": f"Invalid data type: {type(data)}"}
        
        open_time = data.get("open_time", "")
        close_time = data.get("close_time", "")
        timestamp = data.get("timestamp", open_time)
        interval = data.get("interval", "1m")
        open_price = float(data.get("open", 0))
        high_price = float(data.get("high", 0))
        low_price = float(data.get("low", 0))
        close_price = float(data.get("close", 0))
        volume = float(data.get("volume", 0))
        quote_volume = float(data.get("quote_volume", 0))
        trade_count = int(data.get("trade_count", 0))
        taker_buy_base_volume = float(data.get("taker_buy_base_volume", 0))
        taker_buy_quote_volume = float(data.get("taker_buy_quote_volume", 0))
        is_closed = 1  # Always True (already filtered by WebSocket class)
        
        logger.info(f"| 📝 Parsed processed data for {symbol}: timestamp={timestamp}, open={open_price}, close={close_price}")
        
        # Insert candle data
        try:
            # Use uppercase symbol for database
            db_symbol = symbol.upper() if symbol else data.get("symbol", "").upper()
            
            insert_request = InsertRequest(
                table_name=candle_table_name,
                data={
                    "timestamp": timestamp,
                    "open_time": open_time,
                    "close_time": close_time,
                    "symbol": db_symbol,
                    "interval": interval,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "quote_volume": quote_volume,
                    "trade_count": trade_count,
                    "taker_buy_base_volume": taker_buy_base_volume,
                    "taker_buy_quote_volume": taker_buy_quote_volume,
                    "is_closed": is_closed,
                }
            )
            
            logger.info(f"| 💾 Inserting candle data into {candle_table_name} for {db_symbol} (timestamp: {timestamp})")
            result = await self.database_service.insert_data(insert_request)
            
            if result.success:
                logger.info(f"| ✅ Successfully inserted candle data for {db_symbol} (timestamp: {timestamp})")
            else:
                logger.error(f"| ❌ Failed to insert candle data for {db_symbol}: {result.message}")
        except Exception as e:
            logger.error(f"| ❌ Exception in stream_insert for {symbol}: {e}", exc_info=True)
            return {"success": False, "message": str(e)}
        
        if result.success:
            # Only update cache and calculate indicators for closed candles
            if is_closed:
                # Use uppercase symbol for cache and indicators
                cache_symbol = db_symbol
                # Update cache
                await self._update_cache(cache_symbol, {
                    "timestamp": timestamp,
                    "open_time": open_time,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                    "trade_count": trade_count,
                })
                
                # Calculate and insert indicators for closed candle
                await self._calculate_and_insert_indicators(cache_symbol, timestamp, open_time)
        
        return result
    
    async def _update_cache(self, symbol: str, candle_data: Dict) -> None:
        """Update cache with new candle data.
        
        Args:
            symbol: Symbol name
            candle_data: Candle data dictionary
        """
        if symbol not in self._cache:
            self._cache[symbol] = pd.DataFrame()
        
        df = self._cache[symbol]
        new_row = pd.DataFrame([candle_data])
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Limit cache size
        if len(df) > self._cache_limit:
            df = df.tail(self._cache_limit)
        
        self._cache[symbol] = df
    
    async def _calculate_and_insert_indicators(self, symbol: str, timestamp: str, open_time: str) -> None:
        """Calculate technical indicators and insert into database.
        
        Args:
            symbol: Symbol name
            timestamp: Timestamp of the candle ('YYYY-MM-DD HH:MM:SS' format string)
            open_time: Open time ('YYYY-MM-DD HH:MM:SS' format string)
        """
        if symbol not in self._cache or len(self._cache[symbol]) < 2:
            logger.debug(f"| ⚠️  Not enough data for indicators for {symbol}: cache has {len(self._cache.get(symbol, []))} rows")
            return  # Not enough data for indicators
        
        try:
            df = self._cache[symbol].copy()
            # Convert open_time to datetime for proper sorting
            if "open_time" in df.columns:
                df["open_time_dt"] = pd.to_datetime(df["open_time"], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                df = df.sort_values("open_time_dt")
                df = df.drop(columns=["open_time_dt"])
            else:
                df = df.sort_values("open_time")
            
            # Ensure we have numeric columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Convert to numpy arrays
            closes = df["close"].astype(float).values
            highs = df["high"].astype(float).values
            lows = df["low"].astype(float).values
            opens = df["open"].astype(float).values
            volumes = df["volume"].fillna(0.0).astype(float).values
            
            if len(closes) == 0 or len(closes) != len(df):
                return
            
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
                indicators["macd"] = float(macd[-1]) if not pd.isna(macd[-1]) else None
                indicators["macd_signal"] = float(macd_signal[-1]) if not pd.isna(macd_signal[-1]) else None
                indicators["macd_hist"] = float(macd_hist[-1]) if not pd.isna(macd_hist[-1]) else None
            
            # ADX
            if len(highs) >= 14:
                adx = talib.ADX(highs, lows, closes, timeperiod=14)
                indicators["adx"] = float(adx[-1]) if not pd.isna(adx[-1]) else None
            
            # SAR
            if len(highs) >= 2:
                sar = talib.SAR(highs, lows, acceleration=0.02, maximum=0.2)
                indicators["sar"] = float(sar[-1]) if not pd.isna(sar[-1]) else None
            
            # Momentum indicators
            # RSI
            if len(closes) >= 14:
                rsi = talib.RSI(closes, timeperiod=14)
                indicators["rsi"] = float(rsi[-1]) if not pd.isna(rsi[-1]) else None
            
            # Stochastic (KDJ)
            if len(highs) >= 14:
                stoch_k, stoch_d = talib.STOCH(highs, lows, closes, 
                                               fastk_period=14, 
                                               slowk_period=3, 
                                               slowd_period=3)
                indicators["stoch_k"] = float(stoch_k[-1]) if not pd.isna(stoch_k[-1]) else None
                indicators["stoch_d"] = float(stoch_d[-1]) if not pd.isna(stoch_d[-1]) else None
            
            # CCI
            if len(highs) >= 14:
                cci = talib.CCI(highs, lows, closes, timeperiod=14)
                indicators["cci"] = float(cci[-1]) if not pd.isna(cci[-1]) else None
            
            # Volatility indicators
            # Bollinger Bands
            if len(closes) >= 20:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
                indicators["bb_upper"] = float(bb_upper[-1]) if not pd.isna(bb_upper[-1]) else None
                indicators["bb_middle"] = float(bb_middle[-1]) if not pd.isna(bb_middle[-1]) else None
                indicators["bb_lower"] = float(bb_lower[-1]) if not pd.isna(bb_lower[-1]) else None
            
            # ATR
            if len(highs) >= 14:
                atr = talib.ATR(highs, lows, closes, timeperiod=14)
                indicators["atr"] = float(atr[-1]) if not pd.isna(atr[-1]) else None
            
            # Volume indicators
            # OBV
            if len(closes) >= 2 and len(volumes) >= 2:
                obv = talib.OBV(closes, volumes)
                indicators["obv"] = float(obv[-1]) if not pd.isna(obv[-1]) else None
            
            # MFI
            if len(highs) >= 14:
                mfi = talib.MFI(highs, lows, closes, volumes, timeperiod=14)
                indicators["mfi"] = float(mfi[-1]) if not pd.isna(mfi[-1]) else None
            
            # Structure indicators
            # Pivot Points
            if len(df) >= 1:
                current_bar = df.iloc[-1]
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
            
            # Ichimoku
            if len(highs) >= 52:
                tenkan_high = talib.MAX(highs, timeperiod=9)
                tenkan_low = talib.MIN(lows, timeperiod=9)
                tenkan = (tenkan_high + tenkan_low) / 2
                
                kijun_high = talib.MAX(highs, timeperiod=26)
                kijun_low = talib.MIN(lows, timeperiod=26)
                kijun = (kijun_high + kijun_low) / 2
                
                senkou_a = (tenkan + kijun) / 2
                
                senkou_b_high = talib.MAX(highs, timeperiod=52)
                senkou_b_low = talib.MIN(lows, timeperiod=52)
                senkou_b = (senkou_b_high + senkou_b_low) / 2
                
                indicators["ichimoku_tenkan"] = float(tenkan[-1]) if not pd.isna(tenkan[-1]) else None
                indicators["ichimoku_kijun"] = float(kijun[-1]) if not pd.isna(kijun[-1]) else None
                indicators["ichimoku_senkou_a"] = float(senkou_a[-1]) if not pd.isna(senkou_a[-1]) else None
                indicators["ichimoku_senkou_b"] = float(senkou_b[-1]) if not pd.isna(senkou_b[-1]) else None
                if len(closes) >= 26:
                    indicators["ichimoku_chikou"] = float(closes[-26]) if not pd.isna(closes[-26]) else None
            
            # Check if we have any indicators to insert
            if not indicators:
                logger.debug(f"| ⚠️  No indicators calculated for {symbol} at {timestamp}")
                return
            
            # Count non-None indicators
            non_none_count = sum(1 for v in indicators.values() if v is not None)
            logger.debug(f"| 📊 Calculated {non_none_count} non-None indicators for {symbol} at {timestamp} (total: {len(indicators)})")
            
            # Insert indicators
            base_name = self._sanitize_table_name(symbol)
            indicators_table_name = f"{base_name}_indicators"
            
            insert_data = {
                "timestamp": timestamp,
                "open_time": open_time,
                "symbol": symbol,
                **indicators
            }
            
            # Check if indicator record already exists for this timestamp
            check_query = f"SELECT id FROM {indicators_table_name} WHERE timestamp = ? AND symbol = ?"
            check_result = await self.database_service.execute_query(
                QueryRequest(query=check_query, parameters=(timestamp, symbol))
            )
            
            if check_result.success and check_result.extra.get("data"):
                # Update existing record
                record_id = check_result.extra["data"][0]["id"]
                update_fields = ", ".join([f"{k} = ?" for k in insert_data.keys() if k != "timestamp" and k != "symbol" and k != "open_time"])
                update_values = [v for k, v in insert_data.items() if k != "timestamp" and k != "symbol" and k != "open_time"]
                update_values.append(record_id)
                
                update_query = f"UPDATE {indicators_table_name} SET {update_fields} WHERE id = ?"
                update_result = await self.database_service.execute_query(
                    QueryRequest(query=update_query, parameters=tuple(update_values))
                )
                if update_result.success:
                    logger.info(f"| ✅ Updated indicators for {symbol} at {timestamp} ({non_none_count} indicators)")
                else:
                    logger.error(f"| ❌ Failed to update indicators for {symbol} at {timestamp}: {update_result.message}")
            else:
                # Insert new record
                insert_request = InsertRequest(
                    table_name=indicators_table_name,
                    data=insert_data
                )
                result = await self.database_service.insert_data(insert_request)
                if result.success:
                    logger.info(f"| ✅ Inserted indicators for {symbol} at {timestamp} ({non_none_count} indicators)")
                else:
                    logger.error(f"| ❌ Failed to insert indicators for {symbol} at {timestamp}: {result.message}")
            
        except Exception as e:
            logger.warning(f"| ⚠️  Failed to calculate indicators for {symbol} at {timestamp}: {e}", exc_info=True)
    
    async def get_data(self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
        """Get candle data from database with corresponding technical indicators.
        
        Args:
            symbol: Symbol name
            start_date: Optional start date in format 'YYYY-MM-DD HH:MM:SS'
            end_date: Optional end date in format 'YYYY-MM-DD HH:MM:SS'
            limit: Optional limit
            
        Returns:
            List of candle records, each with an 'indicators' field containing non-NULL indicator values
        """
        # Normalize symbol to uppercase for consistency with database
        symbol = symbol.upper() if symbol else ""
        base_name = self._sanitize_table_name(symbol)
        candle_table_name = f"{base_name}_candle"
        indicators_table_name = f"{base_name}_indicators"
        
        # Check if candle table exists
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{candle_table_name}'"
        check_result = await self.database_service.execute_query(
            QueryRequest(query=check_query)
        )
        
        if not check_result.success or not check_result.extra.get("data"):
            logger.warning(f"| ⚠️  Candle table {candle_table_name} does not exist for {symbol}")
            return []
        
        # Build query for candles based on whether date range is provided
        if start_date and end_date:
            query = f"SELECT * FROM {candle_table_name} WHERE symbol = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC, open_time ASC"
            parameters = (symbol, start_date, end_date)
            if limit:
                query += f" LIMIT {limit}"
        else:
            if limit:
                query = f"SELECT * FROM {candle_table_name} WHERE symbol = ? ORDER BY timestamp DESC, open_time DESC LIMIT {limit}"
                parameters = (symbol,)
            else:
                # Get latest timestamp
                max_timestamp_query = f"SELECT MAX(timestamp) as max_ts FROM {candle_table_name} WHERE symbol = ?"
                max_ts_result = await self.database_service.execute_query(
                    QueryRequest(query=max_timestamp_query, parameters=(symbol,))
                )
                
                if not max_ts_result.success or not max_ts_result.extra.get("data"):
                    return []
                
                max_ts_data = max_ts_result.extra.get("data", [])
                if not max_ts_data or not max_ts_data[0].get("max_ts"):
                    return []
                
                latest_timestamp = max_ts_data[0]["max_ts"]
                query = f"SELECT * FROM {candle_table_name} WHERE symbol = ? AND timestamp = ? ORDER BY timestamp ASC, open_time ASC"
                parameters = (symbol, latest_timestamp)
        
        result = await self.database_service.execute_query(
            QueryRequest(query=query, parameters=parameters)
        )
        
        if not result.success:
            logger.warning(f"| ⚠️  Failed to query candles from {candle_table_name}: {result.message}")
            return []
        
        candle_data = result.extra.get("data", [])
        logger.info(f"| 🔍 Query returned {len(candle_data)} rows for {symbol}")
        
        if candle_data:
            logger.debug(f"| 🔍 First row sample: {candle_data[0]}")
        else:
            logger.warning(f"| ⚠️  No candle data returned for {symbol} from table {candle_table_name}")
        
        # If limit was specified and we're not using date range, reverse to get chronological order
        if limit and not start_date and not end_date:
            candle_data.reverse()
        
        # If indicators table exists, fetch indicators for the same timestamps
        check_indicators_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{indicators_table_name}'"
        check_indicators_result = await self.database_service.execute_query(
            QueryRequest(query=check_indicators_query)
        )
        has_indicators_table = check_indicators_result.success and check_indicators_result.extra.get("data")
        
        if has_indicators_table and candle_data:
            # Get all timestamps from candle data
            timestamps = [candle.get("timestamp") for candle in candle_data if candle.get("timestamp")]
            
            indicators_dict = {}
            if timestamps:
                # Build query for indicators
                if start_date and end_date:
                    indicators_query = f"SELECT * FROM {indicators_table_name} WHERE symbol = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
                    indicators_parameters = (symbol, start_date, end_date)
                    if limit:
                        indicators_query += f" LIMIT {limit}"
                else:
                    if limit:
                        indicators_query = f"SELECT * FROM {indicators_table_name} WHERE symbol = ? ORDER BY timestamp DESC, id DESC LIMIT {limit}"
                        indicators_parameters = (symbol,)
                    else:
                        latest_timestamp = timestamps[-1] if timestamps else None
                        if latest_timestamp:
                            indicators_query = f"SELECT * FROM {indicators_table_name} WHERE symbol = ? AND timestamp = ? ORDER BY timestamp ASC, id ASC"
                            indicators_parameters = (symbol, latest_timestamp)
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
                                    "sma_20", "sma_50", "ema_20", "ema_50",
                                    "macd", "macd_signal", "macd_hist",
                                    "adx", "sar", "rsi", "stoch_k", "stoch_d", "cci",
                                    "bb_upper", "bb_middle", "bb_lower", "atr",
                                    "obv", "mfi",
                                    "pivot_point", "pivot_resistance1", "pivot_resistance2",
                                    "pivot_support1", "pivot_support2",
                                    "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
                                    "ichimoku_senkou_b", "ichimoku_chikou"
                                ]
                                
                                for field in indicator_fields:
                                    value = indicator_record.get(field)
                                    if value is not None:
                                        indicators_dict[timestamp][field] = value
            
            # Merge indicators into candle data
            for candle in candle_data:
                timestamp = candle.get("timestamp")
                if timestamp and timestamp in indicators_dict:
                    candle["indicators"] = indicators_dict[timestamp]
                else:
                    candle["indicators"] = {}
        else:
            # No indicators table, add empty indicators dict
            for candle in candle_data:
                candle["indicators"] = {}
        
        return candle_data

