"""News data handler for Alpaca streaming data."""
from typing import Optional, List, Dict
from datetime import datetime, timezone

from src.logger import logger
from src.environment.database.service import DatabaseService
from src.environment.database.types import CreateTableRequest, InsertRequest, QueryRequest
from src.environment.alpacaentry.exceptions import AlpacaError


class NewsHandler:
    """Handler for news data with streaming and caching."""
    
    def __init__(self, database_service: DatabaseService):
        """Initialize news handler.
        
        Args:
            database_service: Database service instance
        """
        self.database_service = database_service
    
    def _sanitize_table_name(self, symbol: Optional[str] = None) -> str:
        """Sanitize symbol name to be used as table name.
        
        For news, we use a global table name regardless of symbol.
        """
        return "data_news"
    
    async def ensure_table_exists(self, symbol: Optional[str] = None) -> None:
        """Ensure news table exists.
        
        Args:
            symbol: Optional symbol name (news table is global, not per-symbol)
        """
        table_name = "data_news_news"
        
        # Check if table already exists
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        check_result = await self.database_service.execute_query(
            QueryRequest(query=check_query)
        )
        
        if check_result.success and check_result.extra.get("data"):
            # Table exists
            return
        
        # Create news table
        columns = [
            {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
            {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
            {"name": "symbol", "type": "TEXT"},
            {"name": "headline", "type": "TEXT"},
            {"name": "summary", "type": "TEXT"},
            {"name": "author", "type": "TEXT"},
            {"name": "source", "type": "TEXT"},
            {"name": "url", "type": "TEXT"},
            {"name": "image_url", "type": "TEXT"},
            {"name": "news_id", "type": "TEXT"},
            {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
        ]
        
        create_request = CreateTableRequest(
            table_name=table_name,
            columns=columns,
            primary_key=None
        )
        result = await self.database_service.create_table(create_request)
        if not result.success:
            logger.error(f"Failed to create news table {table_name}: {result.message}")
            raise AlpacaError(f"Failed to create news table {table_name}: {result.message}")
        
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
    
    def _prepare_data_for_insert(self, data: Dict, symbol: Optional[str] = None) -> Dict:
        """Prepare news data dictionary for database insertion.
        
        Args:
            data: Raw data dictionary from Alpaca stream
            symbol: Optional symbol name
            
        Returns:
            Prepared data dictionary for database insertion
        """
        # Normalize timestamp
        timestamp_value = data.get("timestamp")
        timestamp_str = self._normalize_timestamp(timestamp_value)
        
        # News may have symbols array
        symbols = data.get("symbols", [])
        if symbols:
            symbol_str = ",".join(symbols)  # Join multiple symbols
        else:
            symbol_str = symbol if symbol else ""
        
        db_data = {
            "timestamp": timestamp_str,
            "symbol": symbol_str,
            "headline": data.get("headline"),
            "summary": data.get("summary"),
            "author": data.get("author"),
            "source": data.get("source"),
            "url": data.get("url"),
            "image_url": data.get("image_url"),
            "news_id": data.get("id"),
        }
        
        return db_data
    
    async def stream_insert(self, data: Dict, symbol: Optional[str] = None) -> bool:
        """Insert news data from stream.
        
        Args:
            data: Raw news data dictionary from Alpaca stream
            symbol: Optional symbol name (news may have multiple symbols)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure table exists
            await self.ensure_table_exists(symbol)
            
            # Prepare data for insertion
            db_data = self._prepare_data_for_insert(data, symbol)
            
            # Insert into database
            table_name = "data_news_news"
            
            insert_request = InsertRequest(
                table_name=table_name,
                data=db_data
            )
            
            logger.debug(f"| 🔍 Attempting to insert news data into {table_name}: {db_data}")
            
            result = await self.database_service.insert_data(insert_request)
            
            if not result.success:
                logger.error(f"| ❌ Failed to insert news data: {result.message}")
                logger.error(f"| ❌ Insert request: table={table_name}, data={db_data}")
                return False
            
            # Verify insertion by querying the table
            verify_query = f"SELECT COUNT(*) as count FROM {table_name}"
            verify_result = await self.database_service.execute_query(QueryRequest(query=verify_query))
            if verify_result.success:
                count = verify_result.extra.get("data", [{}])[0].get("count", 0)
                logger.debug(f"| ✅ News data inserted. Total rows in {table_name}: {count}")
            else:
                logger.warning(f"| ⚠️  Insert succeeded but couldn't verify count")
            
            return True
            
        except Exception as e:
            logger.error(f"Error inserting news data: {e}")
            return False
    
    async def get_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get news data from database.
        
        Args:
            symbol: Optional symbol name to filter news (can be comma-separated for multiple symbols)
            start_date: Optional start date in format 'YYYY-MM-DD HH:MM:SS'
            end_date: Optional end date in format 'YYYY-MM-DD HH:MM:SS'
            limit: Optional limit
            
        Returns:
            List of news records
        """
        table_name = "data_news_news"
        
        # Check if news table exists
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        check_result = await self.database_service.execute_query(
            QueryRequest(query=check_query)
        )
        
        if not check_result.success or not check_result.extra.get("data"):
            logger.warning(f"| ⚠️  News table {table_name} does not exist")
            return []
        
        # Debug: Check total row count
        count_query = f"SELECT COUNT(*) as count FROM {table_name}"
        count_result = await self.database_service.execute_query(QueryRequest(query=count_query))
        if count_result.success:
            count = count_result.extra.get("data", [{}])[0].get("count", 0)
            logger.debug(f"| 🔍 Querying news: table {table_name} has {count} rows")
        
        # Build query for news based on whether date range and symbol filter are provided
        where_conditions = []
        parameters = []
        
        if symbol:
            # Symbol may be comma-separated, so we need to check if symbol field contains the symbol
            # Use LIKE for partial matching (since symbol field may contain comma-separated values)
            where_conditions.append("(symbol LIKE ? OR symbol = ?)")
            parameters.append(f"%{symbol}%")
            parameters.append(symbol)
        
        if start_date and end_date:
            where_conditions.append("timestamp >= ?")
            where_conditions.append("timestamp <= ?")
            parameters.append(start_date)
            parameters.append(end_date)
        
        if where_conditions:
            where_clause = " WHERE " + " AND ".join(where_conditions)
        else:
            where_clause = ""
        
        if start_date and end_date:
            query = f"SELECT * FROM {table_name}{where_clause} ORDER BY timestamp ASC"
            if limit:
                query += f" LIMIT {limit}"
        else:
            if limit:
                query = f"SELECT * FROM {table_name}{where_clause} ORDER BY timestamp DESC, id DESC LIMIT {limit}"
            else:
                # Get latest timestamp
                max_timestamp_query = f"SELECT MAX(timestamp) as max_ts FROM {table_name}{where_clause}"
                max_ts_result = await self.database_service.execute_query(
                    QueryRequest(query=max_timestamp_query, parameters=tuple(parameters) if parameters else None)
                )
                
                if not max_ts_result.success or not max_ts_result.extra.get("data"):
                    return []
                
                max_ts_data = max_ts_result.extra.get("data", [])
                if not max_ts_data or not max_ts_data[0].get("max_ts"):
                    return []
                
                latest_timestamp = max_ts_data[0]["max_ts"]
                where_conditions.append("timestamp = ?")
                parameters.append(latest_timestamp)
                where_clause = " WHERE " + " AND ".join(where_conditions)
                query = f"SELECT * FROM {table_name}{where_clause} ORDER BY timestamp ASC, id ASC"
        
        # Debug: Log the query being executed
        logger.debug(f"| 🔍 Executing query for news: {query}")
        if parameters:
            logger.debug(f"| 🔍 Query parameters: {parameters}")
        
        result = await self.database_service.execute_query(
            QueryRequest(query=query, parameters=tuple(parameters) if parameters else None)
        )
        
        if not result.success:
            logger.warning(f"| ⚠️  Failed to query news from {table_name}: {result.message}")
            return []
        
        news_data = result.extra.get("data", [])
        logger.debug(f"| 🔍 Query returned {len(news_data)} rows for news")
        
        if news_data:
            logger.debug(f"| 🔍 First row sample: {news_data[0]}")
        else:
            logger.warning(f"| ⚠️  No news data returned from table {table_name}")
        
        # If limit was specified and we're not using date range, reverse to get chronological order
        if limit and not start_date and not end_date:
            news_data.reverse()
        
        return news_data
