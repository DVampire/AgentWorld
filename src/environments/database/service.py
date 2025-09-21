"""Database service using aiosqlite for async database operations."""

import asyncio
import aiosqlite
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.environments.database.types import (
    QueryRequest, 
    QueryResult, 
    TableInfo,
    DatabaseInfo,
    CreateTableRequest, 
    InsertRequest,
    UpdateRequest,
    DeleteRequest,
    SelectRequest, 
    SelectResult, 
    GetTablesRequest,
    GetTablesResult
)
from src.environments.database.exceptions import (
    DatabaseError, 
    InvalidQueryError, 
    TableNotFoundError,
    ColumnNotFoundError,
    ConstraintViolationError, 
    ConnectionError
)


class DatabaseService:
    """Async database service using aiosqlite."""
    
    def __init__(self, db_path: Union[str, Path]):
        """Initialize the database service.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
        self._connection: Optional[aiosqlite.Connection] = None
        self._is_connected = False
    
    async def connect(self) -> None:
        """Connect to the database."""
        try:
            # Ensure the directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            self._connection = await aiosqlite.connect(str(self.db_path))
            self._is_connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._is_connected = False
    
    async def execute_query(self, request: QueryRequest) -> QueryResult:
        """Execute a SQL query.
        
        Args:
            request: Query request with SQL and parameters
            
        Returns:
            Query result with data and metadata
        """
        if not self._is_connected:
            raise ConnectionError("Database not connected")
        
        start_time = time.time()
        
        try:
            cursor = await self._connection.execute(request.query, request.parameters or {})
            
            # Check if it's a SELECT query
            if request.query.strip().upper().startswith('SELECT'):
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description] if cursor.description else []
                
                # Convert rows to dictionaries
                data = [dict(zip(columns, row)) for row in rows]
                
                await cursor.close()
                await self._connection.commit()
                
                execution_time = time.time() - start_time
                return QueryResult(
                    success=True,
                    data=data,
                    row_count=len(data),
                    message=f"Query executed successfully, returned {len(data)} rows",
                    execution_time=execution_time
                )
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
                await self._connection.commit()
                await cursor.close()
                
                execution_time = time.time() - start_time
                return QueryResult(
                    success=True,
                    data=None,
                    row_count=cursor.rowcount,
                    message=f"Query executed successfully, affected {cursor.rowcount} rows",
                    execution_time=execution_time
                )
                
        except Exception as e:
            await self._connection.rollback()
            execution_time = time.time() - start_time
            return QueryResult(
                success=False,
                data=None,
                row_count=0,
                message=f"Query failed: {str(e)}",
                execution_time=execution_time
            )
    
    async def create_table(self, request: CreateTableRequest) -> QueryResult:
        """Create a table.
        
        Args:
            request: Table creation request
            
        Returns:
            Query result
        """
        # Build CREATE TABLE SQL
        columns_sql = []
        for col in request.columns:
            col_name = col['name']
            col_type = col.get('type', 'TEXT')
            col_constraints = col.get('constraints', '')
            
            column_sql = f"{col_name} {col_type}"
            if col_constraints:
                column_sql += f" {col_constraints}"
            columns_sql.append(column_sql)
        
        # Add primary key if specified
        if request.primary_key:
            columns_sql.append(f"PRIMARY KEY ({request.primary_key})")
        
        # Add foreign keys if specified
        if request.foreign_keys:
            for fk in request.foreign_keys:
                fk_sql = f"FOREIGN KEY ({fk['column']}) REFERENCES {fk['ref_table']}({fk['ref_column']})"
                columns_sql.append(fk_sql)
        
        sql = f"CREATE TABLE IF NOT EXISTS {request.table_name} ({', '.join(columns_sql)})"
        
        query_request = QueryRequest(query=sql)
        return await self.execute_query(query_request)
    
    async def insert_data(self, request: InsertRequest) -> QueryResult:
        """Insert data into a table.
        
        Args:
            request: Insert request
            
        Returns:
            Query result
        """
        if isinstance(request.data, dict):
            # Single row insert
            columns = list(request.data.keys())
            placeholders = [f":{col}" for col in columns]
            
            sql = f"INSERT INTO {request.table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            query_request = QueryRequest(query=sql, parameters=request.data)
            
        else:
            # Multiple rows insert
            if not request.data:
                raise InvalidQueryError("No data provided for insert")
            
            columns = list(request.data[0].keys())
            placeholders = [f":{col}" for col in columns]
            
            sql = f"INSERT INTO {request.table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            
            # Execute multiple inserts
            results = []
            for row in request.data:
                query_request = QueryRequest(query=sql, parameters=row)
                result = await self.execute_query(query_request)
                results.append(result)
            
            # Return summary result
            total_affected = sum(r.row_count for r in results)
            return QueryResult(
                success=all(r.success for r in results),
                data=None,
                row_count=total_affected,
                message=f"Inserted {total_affected} rows",
                execution_time=sum(r.execution_time or 0 for r in results)
            )
        
        return await self.execute_query(query_request)
    
    async def update_data(self, request: UpdateRequest) -> QueryResult:
        """Update data in a table.
        
        Args:
            request: Update request
            
        Returns:
            Query result
        """
        set_clauses = [f"{col} = :{col}" for col in request.data.keys()]
        sql = f"UPDATE {request.table_name} SET {', '.join(set_clauses)} WHERE {request.where_clause}"
        
        # Combine data and where parameters
        parameters = {**request.data, **(request.where_params or {})}
        
        query_request = QueryRequest(query=sql, parameters=parameters)
        return await self.execute_query(query_request)
    
    async def delete_data(self, request: DeleteRequest) -> QueryResult:
        """Delete data from a table.
        
        Args:
            request: Delete request
            
        Returns:
            Query result
        """
        sql = f"DELETE FROM {request.table_name} WHERE {request.where_clause}"
        query_request = QueryRequest(query=sql, parameters=request.where_params or {})
        return await self.execute_query(query_request)
    
    async def select_data(self, request: SelectRequest) -> SelectResult:
        """Select data from a table.
        
        Args:
            request: Select request with table name, columns, where clause, etc.
            
        Returns:
            Select result with data and metadata
        """
        # Build SELECT query
        columns_str = "*" if not request.columns else ", ".join(request.columns)
        sql = f"SELECT {columns_str} FROM {request.table_name}"
        
        # Add WHERE clause if provided
        if request.where_clause:
            sql += f" WHERE {request.where_clause}"
        
        # Add ORDER BY clause if provided
        if request.order_by:
            sql += f" ORDER BY {request.order_by}"
        
        # Add LIMIT clause if provided
        if request.limit:
            sql += f" LIMIT {request.limit}"
        
        query_request = QueryRequest(query=sql, parameters=request.where_params or {})
        result = await self.execute_query(query_request)
        
        return SelectResult(
            success=result.success,
            data=result.data,
            row_count=result.row_count,
            message=result.message,
            execution_time=result.execution_time
        )
    
    async def get_tables(self, request: GetTablesRequest = None) -> GetTablesResult:
        """Get information about all tables in the database.
        
        Returns:
            List of table information
        """
        # Get list of tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = await self.execute_query(QueryRequest(query=tables_query))
        
        try:
            if not tables_result.success or not tables_result.data:
                return GetTablesResult(
                    success=True,
                    tables=[],
                    message="No tables found"
                )
            
            tables = []
            for table_row in tables_result.data:
                table_name = table_row['name']
                
                # Get table schema
                schema_query = f"PRAGMA table_info({table_name})"
                schema_result = await self.execute_query(QueryRequest(query=schema_query))
                
                columns = schema_result.data if schema_result.success else []
                
                # Get row count
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_result = await self.execute_query(QueryRequest(query=count_query))
                row_count = count_result.data[0]['count'] if count_result.success and count_result.data else 0
                
                tables.append(TableInfo(
                    name=table_name,
                    columns=columns,
                    row_count=row_count
                ))
            
            return GetTablesResult(
                success=True,
                tables=tables,
                message=f"Found {len(tables)} tables"
            )
        except Exception as e:
            return GetTablesResult(
                success=False,
                tables=[],
                message=f"Failed to get tables: {str(e)}"
            )
    
    async def get_database_info(self) -> DatabaseInfo:
        """Get information about the database.
        
        Returns:
            Database information
        """
        tables = await self.get_tables()
        
        return DatabaseInfo(
            path=str(self.db_path),
            tables=tables,
            total_tables=len(tables),
            is_connected=self._is_connected
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
