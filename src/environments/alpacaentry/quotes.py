"""Quotes data handler for Alpaca streaming data."""
from typing import Optional, List, Dict

from src.logger import logger
from src.environments.database.service import DatabaseService
from src.environments.database.types import CreateTableRequest, InsertRequest, QueryRequest
from src.environments.alpacaentry.exceptions import AlpacaError


class QuotesHandler:
    """Handler for quotes data with streaming and caching.
    
    TODO: Implement streaming, caching, and data retrieval methods.
    """
    
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
    
    async def ensure_table_exists(self, symbol: str, asset_type: str = "crypto") -> None:
        """Ensure quotes table exists for a symbol.
        
        Args:
            symbol: Symbol name
            asset_type: Asset type ("crypto" or "stock")
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
        
        # Create quotes table based on asset type
        if asset_type == "crypto":
            columns = [
                {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
                {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "bid_price", "type": "REAL"},
                {"name": "bid_size", "type": "REAL"},
                {"name": "ask_price", "type": "REAL"},
                {"name": "ask_size", "type": "REAL"},
                {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
            ]
        else:  # stock
            columns = [
                {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
                {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
                {"name": "bid_price", "type": "REAL"},
                {"name": "bid_size", "type": "REAL"},
                {"name": "ask_price", "type": "REAL"},
                {"name": "ask_size", "type": "REAL"},
                {"name": "tape", "type": "TEXT"},
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
        index_name = f"{table_name}_timestamp_idx"
        index_query = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}(timestamp DESC)"
        index_result = await self.database_service.execute_query(QueryRequest(query=index_query))
        if not index_result.success:
            logger.warning(f"Failed to create index {index_name} for {table_name}: {index_result.message}")
    
    async def stream_insert(self, data: Dict, symbol: str, asset_type: str = "crypto") -> bool:
        """Insert quotes data from stream.
        
        Args:
            data: Raw quotes data from Alpaca stream
            symbol: Symbol name
            asset_type: Asset type ("crypto" or "stock")
            
        Returns:
            True if successful, False otherwise
            
        TODO: Implement streaming insert with caching.
        """
        # TODO: Implement streaming insert
        raise NotImplementedError("Quotes streaming insert not yet implemented")
    
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
            
        TODO: Implement data retrieval.
        """
        # TODO: Implement data retrieval
        raise NotImplementedError("Quotes data retrieval not yet implemented")

