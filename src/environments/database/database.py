"""Database class for AgentWorld - provides database operations interface."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .service import DatabaseService
from .types import (
    QueryRequest, QueryResult, TableInfo, DatabaseInfo,
    CreateTableRequest, InsertRequest, UpdateRequest, DeleteRequest
)
from .exceptions import DatabaseError, ConnectionError
from src.utils import assemble_project_path


class Database:
    """Database interface for AgentWorld with async operations."""
    
    def __init__(self, db_path: Union[str, Path], auto_connect: bool = True):
        """Initialize the database.
        
        Args:
            db_path: Path to the SQLite database file
            auto_connect: Whether to automatically connect to the database
        """
        self.db_path = Path(assemble_project_path(str(db_path))) if isinstance(db_path, str) else db_path
        self.auto_connect = auto_connect
        
        # Initialize database service
        self._service = DatabaseService(self.db_path)
        self._is_connected = False
    
    async def connect(self) -> None:
        """Connect to the database."""
        await self._service.connect()
        self._is_connected = True
    
    async def disconnect(self) -> None:
        """Disconnect from the database."""
        await self._service.disconnect()
        self._is_connected = False
    
    async def ensure_connected(self) -> None:
        """Ensure database is connected."""
        if not self._is_connected and self.auto_connect:
            await self.connect()
        elif not self._is_connected:
            raise ConnectionError("Database not connected and auto_connect is disabled")
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> QueryResult:
        """Execute a SQL query.
        
        Args:
            query: SQL query string
            parameters: Query parameters
            
        Returns:
            Query result
        """
        await self.ensure_connected()
        request = QueryRequest(query=query, parameters=parameters)
        return await self._service.execute_query(request)
    
    async def create_table(
        self,
        table_name: str,
        columns: List[Dict[str, Any]],
        primary_key: Optional[str] = None,
        foreign_keys: Optional[List[Dict[str, Any]]] = None
    ) -> QueryResult:
        """Create a table.
        
        Args:
            table_name: Name of the table
            columns: List of column definitions
            primary_key: Primary key column name
            foreign_keys: List of foreign key constraints
            
        Returns:
            Query result
        """
        await self.ensure_connected()
        request = CreateTableRequest(
            table_name=table_name,
            columns=columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys
        )
        return await self._service.create_table(request)
    
    async def insert(
        self,
        table_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> QueryResult:
        """Insert data into a table.
        
        Args:
            table_name: Name of the table
            data: Data to insert (single row or multiple rows)
            
        Returns:
            Query result
        """
        await self.ensure_connected()
        request = InsertRequest(table_name=table_name, data=data)
        return await self._service.insert_data(request)
    
    async def update(
        self,
        table_name: str,
        data: Dict[str, Any],
        where_clause: str,
        where_params: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Update data in a table.
        
        Args:
            table_name: Name of the table
            data: Data to update
            where_clause: WHERE clause
            where_params: Parameters for WHERE clause
            
        Returns:
            Query result
        """
        await self.ensure_connected()
        request = UpdateRequest(
            table_name=table_name,
            data=data,
            where_clause=where_clause,
            where_params=where_params
        )
        return await self._service.update_data(request)
    
    async def delete(
        self,
        table_name: str,
        where_clause: str,
        where_params: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """Delete data from a table.
        
        Args:
            table_name: Name of the table
            where_clause: WHERE clause
            where_params: Parameters for WHERE clause
            
        Returns:
            Query result
        """
        await self.ensure_connected()
        request = DeleteRequest(
            table_name=table_name,
            where_clause=where_clause,
            where_params=where_params
        )
        return await self._service.delete_data(request)
    
    async def select(
        self,
        table_name: str,
        columns: Optional[List[str]] = None,
        where_clause: Optional[str] = None,
        where_params: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None
    ) -> QueryResult:
        """Select data from a table.
        
        Args:
            table_name: Name of the table
            columns: List of columns to select (None for all)
            where_clause: WHERE clause
            where_params: Parameters for WHERE clause
            order_by: ORDER BY clause
            limit: LIMIT clause
            
        Returns:
            Query result
        """
        await self.ensure_connected()
        
        # Build SELECT query
        columns_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {columns_str} FROM {table_name}"
        
        parameters = where_params or {}
        
        if where_clause:
            query += f" WHERE {where_clause}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            query += f" LIMIT {limit}"
        
        return await self.execute_query(query, parameters)
    
    async def get_tables(self) -> List[TableInfo]:
        """Get information about all tables.
        
        Returns:
            List of table information
        """
        await self.ensure_connected()
        return await self._service.get_tables()
    
    async def get_database_info(self) -> DatabaseInfo:
        """Get information about the database.
        
        Returns:
            Database information
        """
        await self.ensure_connected()
        return await self._service.get_database_info()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
