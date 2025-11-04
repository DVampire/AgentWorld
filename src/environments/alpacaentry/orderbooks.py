"""Orderbooks data handler for Alpaca streaming data."""
from typing import Optional, List, Dict

from src.logger import logger
from src.environments.database.service import DatabaseService
from src.environments.database.types import CreateTableRequest, InsertRequest, QueryRequest
from src.environments.alpacaentry.exceptions import AlpacaError


class OrderbooksHandler:
    """Handler for orderbooks data with streaming and caching.
    
    Note: Orderbooks are only available for crypto assets, not stocks.
    
    TODO: Implement streaming, caching, and data retrieval methods.
    """
    
    def __init__(self, database_service: DatabaseService):
        """Initialize orderbooks handler.
        
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
        """Ensure orderbooks table exists for a symbol.
        
        Args:
            symbol: Symbol name (crypto only)
        """
        base_name = self._sanitize_table_name(symbol)
        table_name = f"{base_name}_orderbooks"
        
        # Check if table already exists
        check_query = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        check_result = await self.database_service.execute_query(
            QueryRequest(query=check_query)
        )
        
        if check_result.success and check_result.extra.get("data"):
            # Table exists
            return
        
        # Create orderbooks table
        columns = [
            {"name": "id", "type": "INTEGER", "constraints": "PRIMARY KEY AUTOINCREMENT"},
            {"name": "timestamp", "type": "TEXT", "constraints": "NOT NULL"},
            {"name": "symbol", "type": "TEXT", "constraints": "NOT NULL"},
            {"name": "bids", "type": "TEXT"},  # JSON string for bid arrays
            {"name": "asks", "type": "TEXT"},  # JSON string for ask arrays
            {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
        ]
        
        create_request = CreateTableRequest(
            table_name=table_name,
            columns=columns,
            primary_key=None
        )
        result = await self.database_service.create_table(create_request)
        if not result.success:
            logger.error(f"Failed to create orderbooks table {table_name}: {result.message}")
            raise AlpacaError(f"Failed to create orderbooks table {table_name}: {result.message}")
    
    async def stream_insert(self, data: Dict, symbol: str) -> bool:
        """Insert orderbooks data from stream.
        
        Args:
            data: Raw orderbooks data from Alpaca stream
            symbol: Symbol name (crypto only)
            
        Returns:
            True if successful, False otherwise
            
        TODO: Implement streaming insert with caching.
        """
        # TODO: Implement streaming insert
        raise NotImplementedError("Orderbooks streaming insert not yet implemented")
    
    async def get_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get orderbooks data from database.
        
        Args:
            symbol: Symbol name
            start_date: Optional start date in format 'YYYY-MM-DD HH:MM:SS'
            end_date: Optional end date in format 'YYYY-MM-DD HH:MM:SS'
            limit: Optional limit
            
        Returns:
            List of orderbooks records
            
        TODO: Implement data retrieval.
        """
        # TODO: Implement data retrieval
        raise NotImplementedError("Orderbooks data retrieval not yet implemented")

