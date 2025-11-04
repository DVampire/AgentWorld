"""Database service using aiosqlite for async database operations."""

import aiosqlite
import time
from pathlib import Path
from typing import Optional, Union

from src.environments.protocol.types import ActionResult
from src.environments.database.types import (
    QueryRequest, 
    TableInfo,
    CreateTableRequest, 
    InsertRequest,
    UpdateRequest,
    DeleteRequest,
    SelectRequest, 
    GetTablesRequest
)
from src.environments.database.exceptions import (
    ConnectionError
)


class DatabaseService:
    """Async database service using aiosqlite."""
    
    def __init__(self, base_dir: Union[str, Path]):
        """Initialize the database service.
        
        Args:
            base_dir: Base directory for the database
        """
        self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
        self._connection: Optional[aiosqlite.Connection] = None
        self._is_connected = False
    
    async def connect(self) -> None:
        """Connect to the database."""
        try:
            # Ensure the directory exists
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
            self._connection = await aiosqlite.connect(str(self.base_dir / "database.db"))
            self._is_connected = True
        except Exception as e:
            raise ConnectionError(f"Failed to connect to database: {e}")
    
    async def disconnect(self) -> None:
        """Disconnect from the database."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._is_connected = False
    
    async def execute_query(self, request: QueryRequest) -> ActionResult:
        """Execute a SQL query.
        
        Args:
            request: Query request with SQL and parameters
            
        Returns:
            Action result with data and metadata in extra
        """
        if not self._is_connected:
            return ActionResult(
                success=False,
                message="Database not connected",
                extra={"error": "Database not connected"}
            )
        
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
                return ActionResult(
                    success=True,
                    message=f"Query executed successfully, returned {len(data)} rows",
                    extra={
                        "data": data,
                        "row_count": len(data),
                        "execution_time": execution_time,
                        "query": request.query
                    }
                )
            else:
                # For non-SELECT queries (INSERT, UPDATE, DELETE, etc.)
                await self._connection.commit()
                rowcount = cursor.rowcount
                await cursor.close()
                
                execution_time = time.time() - start_time
                return ActionResult(
                    success=True,
                    message=f"Query executed successfully, affected {rowcount} rows",
                    extra={
                        "row_count": rowcount,
                        "execution_time": execution_time,
                        "query": request.query
                    }
                )
                
        except Exception as e:
            await self._connection.rollback()
            execution_time = time.time() - start_time
            return ActionResult(
                success=False,
                message=f"Query failed: {str(e)}",
                extra={
                    "error": str(e),
                    "execution_time": execution_time,
                    "query": request.query
                }
            )
    
    async def create_table(self, request: CreateTableRequest) -> ActionResult:
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
        result = await self.execute_query(query_request)
        # Add table_name and table info to extra
        if result.extra:
            result.extra["table_name"] = request.table_name
            result.extra["columns"] = request.columns
            result.extra["primary_key"] = request.primary_key
            result.extra["foreign_keys"] = request.foreign_keys
        return result
    
    async def insert_data(self, request: InsertRequest) -> ActionResult:
        """Insert data into a table.
        
        Args:
            request: Insert request
            
        Returns:
            Action result with insert information in extra
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
                return ActionResult(
                    success=False,
                    message="No data provided for insert",
                    extra={"error": "No data provided for insert"}
                )
            
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
            total_affected = sum(r.extra.get("row_count", 0) for r in results if r.success)
            total_time = sum(r.extra.get("execution_time", 0) for r in results)
            all_success = all(r.success for r in results)
            
            return ActionResult(
                success=all_success,
                message=f"Inserted {total_affected} rows",
                extra={
                    "row_count": total_affected,
                    "execution_time": total_time,
                    "table_name": request.table_name,
                    "num_rows_inserted": len(request.data)
                }
            )
        
        result = await self.execute_query(query_request)
        # Add table_name to extra
        if result.extra:
            result.extra["table_name"] = request.table_name
        return result
    
    async def update_data(self, request: UpdateRequest) -> ActionResult:
        """Update data in a table.
        
        Args:
            request: Update request
            
        Returns:
            Action result with update information in extra
        """
        set_clauses = [f"{col} = :{col}" for col in request.data.keys()]
        sql = f"UPDATE {request.table_name} SET {', '.join(set_clauses)} WHERE {request.where_clause}"
        
        # Combine data and where parameters
        parameters = {**request.data, **(request.where_params or {})}
        
        query_request = QueryRequest(query=sql, parameters=parameters)
        result = await self.execute_query(query_request)
        # Add table_name to extra
        if result.extra:
            result.extra["table_name"] = request.table_name
        return result
    
    async def delete_data(self, request: DeleteRequest) -> ActionResult:
        """Delete data from a table.
        
        Args:
            request: Delete request
            
        Returns:
            Action result with delete information in extra
        """
        sql = f"DELETE FROM {request.table_name} WHERE {request.where_clause}"
        query_request = QueryRequest(query=sql, parameters=request.where_params or {})
        result = await self.execute_query(query_request)
        # Add table_name to extra
        if result.extra:
            result.extra["table_name"] = request.table_name
        return result
    
    async def select_data(self, request: SelectRequest) -> ActionResult:
        """Select data from a table.
        
        Args:
            request: Select request with table name, columns, where clause, etc.
            
        Returns:
            Action result with data and metadata in extra
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
        # Add table_name and query details to extra
        if result.extra:
            result.extra["table_name"] = request.table_name
            result.extra["columns"] = request.columns
        return result
    
    async def get_tables(self, request: GetTablesRequest = None) -> ActionResult:
        """Get information about all tables in the database.
        
        Returns:
            Action result with table information in extra
        """
        # Get list of tables
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        tables_result = await self.execute_query(QueryRequest(query=tables_query))
        
        try:
            if not tables_result.success or not tables_result.extra or not tables_result.extra.get("data"):
                return ActionResult(
                    success=True,
                    message="No tables found",
                    extra={
                        "tables": [],
                        "total_tables": 0
                    }
                )
            
            tables_data = tables_result.extra.get("data", [])
            tables = []
            for table_row in tables_data:
                table_name = table_row['name']
                
                # Get table schema
                schema_query = f"PRAGMA table_info({table_name})"
                schema_result = await self.execute_query(QueryRequest(query=schema_query))
                
                columns = schema_result.extra.get("data", []) if schema_result.success else []
                
                # Get row count
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_result = await self.execute_query(QueryRequest(query=count_query))
                row_count = count_result.extra.get("data", [{}])[0].get('count', 0) if count_result.success and count_result.extra.get("data") else 0
                
                tables.append(TableInfo(
                    name=table_name,
                    columns=columns,
                    row_count=row_count
                ))
            
            # Convert TableInfo to dict for extra
            tables_dict = [table.model_dump() for table in tables]
            
            return ActionResult(
                success=True,
                message=f"Found {len(tables)} tables",
                extra={
                    "tables": tables_dict,
                    "total_tables": len(tables)
                }
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to get tables: {str(e)}",
                extra={
                    "error": str(e),
                    "tables": [],
                    "total_tables": 0
                }
            )
    
    async def get_database_info(self) -> ActionResult:
        """Get information about the database.
        
        Returns:
            Database information
        """
        tables = await self.get_tables()
        
        return ActionResult(
            success=True,
            message=f"Database information retrieved successfully",
            extra={
                "path": str(self.base_dir / "database.db"),
                "tables": tables,
                "total_tables": len(tables),
                "is_connected": self._is_connected
            }
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
