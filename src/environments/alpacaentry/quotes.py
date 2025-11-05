"""Quotes data handler for Alpaca streaming data."""
from typing import Optional, List, Dict
from datetime import datetime, timezone
from alpaca.trading.enums import AssetClass

from src.logger import logger
from src.environments.database.service import DatabaseService
from src.environments.database.types import CreateTableRequest, InsertRequest, QueryRequest
from src.environments.alpacaentry.exceptions import AlpacaError


class QuotesHandler:
    """Handler for quotes data with streaming and caching."""
    
    def __init__(self, database_service: DatabaseService):
        """Initialize quotes handler.
        
        Args:
            database_service: Database service instance
        """
        self.database_service = database_service
    
    def _sanitize_table_name(self, symbol: str) -> str:
        """Sanitize symbol name to be used as table name."""
        # Replace invalid characters with underscore
        table_name = symbol.replace("/", "_").replace(".", "_").replace("-", "_")
        # Remove any other invalid characters
        table_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)
        return f"data_{table_name}"
    
    async def ensure_table_exists(self, symbol: str) -> None:
        """Ensure quotes table exists for a symbol.
        
        Args:
            symbol: Symbol name
        """
        base_name = self._sanitize_table_name(symbol)
        table_name = f"{base_name}_quotes"
        
        # Check if table already exists
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        check_result = await self.database_service.execute_query(
            QueryRequest(query=check_query)
        )
        
        if check_result.success and check_result.extra.get("data"):
            # Table exists
            return
        
        # Create quotes table (supports both crypto and stock by including optional tape field)
        columns = [
            {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
            {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
            {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
            {"name": "bid_price", "type": "REAL"},
            {"name": "bid_size", "type": "REAL"},
            {"name": "ask_price", "type": "REAL"},
            {"name": "ask_size", "type": "REAL"},
            {"name": "tape", "type": "TEXT"},  # Optional, for stocks
            {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
        ]
        
        create_request = CreateTableRequest(
            table_name=table_name,
            columns=columns,
            primary_key=None
        )
        result = await self.database_service.create_table(create_request)
        if not result.success:
            logger.error(f"Failed to create quotes table {table_name}: {result.message}")
            raise AlpacaError(f"Failed to create quotes table {table_name}: {result.message}")
        
        # Create index for performance optimization
        index_name = f"{table_name}_timestamp_id_idx"
        index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}(timestamp DESC, id DESC)"
        index_result = await self.database_service.execute_query(QueryRequest(query=index_query))
        if not index_result.success:
            logger.warning(f"Failed to create index {index_name} for {table_name}: {index_result.message}")
    
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
    
    def _prepare_data_for_insert(self, data: Dict, symbol: str, asset_type: AssetClass = AssetClass.CRYPTO) -> Dict:
        """Prepare quotes data dictionary for database insertion.
        
        Args:
            data: Raw data dictionary from Alpaca stream
            symbol: Symbol name
            asset_type: Asset class (AssetClass.CRYPTO or AssetClass.US_EQUITY)
            
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
            "bid_price": data.get("bid_price") if "bid_price" in data else data.get("bp"),
            "bid_size": data.get("bid_size") if "bid_size" in data else data.get("bs"),
            "ask_price": data.get("ask_price") if "ask_price" in data else data.get("ap"),
            "ask_size": data.get("ask_size") if "ask_size" in data else data.get("as"),
        }
        
        # Add tape field for stocks
        if asset_type == AssetClass.US_EQUITY:
            db_data["tape"] = data.get("tape")
        
        return db_data
    
    async def stream_insert(self, data: Dict, symbol: str, asset_type: AssetClass = AssetClass.CRYPTO) -> bool:
        """Insert quotes data from stream.
        
        Args:
            data: Raw quotes data dictionary from Alpaca stream
            symbol: Symbol name
            asset_type: Asset class (AssetClass.CRYPTO or AssetClass.US_EQUITY)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure table exists
            await self.ensure_table_exists(symbol)
            
            # Prepare data for insertion
            db_data = self._prepare_data_for_insert(data, symbol, asset_type)
            
            # Insert into database
            base_name = self._sanitize_table_name(symbol)
            table_name = f"{base_name}_quotes"
            
            insert_request = InsertRequest(
                table_name=table_name,
                data=db_data
            )
            
            logger.debug(f"| 🔍 Attempting to insert quotes data for {symbol} into {table_name}: {db_data}")
            
            result = await self.database_service.insert_data(insert_request)
            
            if not result.success:
                logger.error(f"| ❌ Failed to insert quotes data for {symbol}: {result.message}")
                logger.error(f"| ❌ Insert request: table={table_name}, data={db_data}")
                return False
            
            # Verify insertion by querying the table
            verify_query = f"SELECT COUNT(*) as count FROM {table_name}"
            verify_result = await self.database_service.execute_query(QueryRequest(query=verify_query))
            if verify_result.success:
                count = verify_result.extra.get("data", [{}])[0].get("count", 0)
                logger.debug(f"| ✅ Quotes data inserted for {symbol}. Total rows in {table_name}: {count}")
            else:
                logger.warning(f"| ⚠️  Insert succeeded but couldn't verify count for {symbol}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error inserting quotes data for {symbol}: {e}")
            return False
    
    async def get_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get quotes data from database.
        
        Args:
            symbol: Symbol name
            start_date: Optional start date in format 'YYYY-MM-DD HH:MM:SS'
            end_date: Optional end date in format 'YYYY-MM-DD HH:MM:SS'
            limit: Optional limit
            
        Returns:
            List of quotes records
        """
        base_name = self._sanitize_table_name(symbol)
        table_name = f"{base_name}_quotes"
        
        # Check if quotes table exists
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        check_result = await self.database_service.execute_query(
            QueryRequest(query=check_query)
        )
        
        if not check_result.success or not check_result.extra.get("data"):
            logger.warning(f"| ⚠️  Quotes table {table_name} does not exist for {symbol}")
            return []
        
        # Debug: Check total row count
        count_query = f"SELECT COUNT(*) as count FROM {table_name}"
        count_result = await self.database_service.execute_query(QueryRequest(query=count_query))
        if count_result.success:
            count = count_result.extra.get("data", [{}])[0].get("count", 0)
            logger.debug(f"| 🔍 Querying quotes for {symbol}: table {table_name} has {count} rows")
        
        # Build query for quotes based on whether date range is provided
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
            logger.warning(f"| ⚠️  Failed to query quotes from {table_name}: {result.message}")
            return []
        
        quotes_data = result.extra.get("data", [])
        logger.debug(f"| 🔍 Query returned {len(quotes_data)} rows for {symbol}")
        
        if quotes_data:
            logger.debug(f"| 🔍 First row sample: {quotes_data[0]}")
        else:
            logger.warning(f"| ⚠️  No quotes data returned for {symbol} from table {table_name}")
        
        # If limit was specified and we're not using date range, reverse to get chronological order
        if limit and not start_date and not end_date:
            quotes_data.reverse()
        
        return quotes_data
