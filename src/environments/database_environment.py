"""Database Environment for AgentWorld - provides database operations as an environment."""

import os
from typing import Any, Dict, List, Union, Optional, Type
from pydantic import BaseModel, Field, ConfigDict

from src.environments.database.service import DatabaseService
from src.environments.database.types import (
    QueryRequest, 
    CreateTableRequest, 
    InsertRequest,
    UpdateRequest, 
    DeleteRequest,
    SelectRequest, 
    GetTablesRequest
)
from src.logger import logger
from src.utils import assemble_project_path
from src.environments.protocol.server import ecp
from src.environments.protocol.environment import BaseEnvironment

@ecp.environment()
class DatabaseEnvironment(BaseEnvironment):
    """Database Environment that provides database operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="database", description="The name of the database environment.")
    type: str = Field(default="Database", description="The type of the database environment.")
    description: str = Field(default="Database environment for SQLite database operations", description="The description of the database environment.")
    args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the database environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the database environment including tables and data.",
        }
    }, description="The metadata of the database environment.")
    
    def __init__(
        self,
        base_dir: str,
        auto_connect: bool = True,
        create_sample_tables: bool = True,
        **kwargs
    ):
        """
        Initialize the database environment.
        
        Args:
            base_dir (str): Base directory for database storage
            auto_connect (bool): Whether to automatically connect to the database
            create_sample_tables (bool): Whether to create sample tables for testing
        """
        super().__init__(**kwargs)
        
        self.base_dir = assemble_project_path(base_dir)
        self.auto_connect = auto_connect
        self.create_sample_tables = create_sample_tables
        
        # Initialize database service
        self.database_service = DatabaseService(db_path=os.path.join(self.base_dir, "database.db"))
        
        
    async def initialize(self) -> None:
        """Initialize the database environment."""
        logger.info(f"| 🗄️ Database Environment initialized at: {self.base_dir}")
        
        # Connect to database if auto_connect is enabled
        if self.auto_connect:
            await self.database_service.connect()
        
        if self.create_sample_tables:
            await self._create_sample_tables()
    
    async def cleanup(self) -> None:
        """Cleanup the database environment."""
        await self.database_service.disconnect()
        logger.info("| 🗄️ Database Environment cleaned up")
    
    async def _create_sample_tables(self) -> None:
        """Create sample tables for testing."""
        try:
            # Create users table
            create_users_request = CreateTableRequest(
                table_name="users",
                columns=[
                    {"name": "id", "type": "INTEGER", "constraints": "NOT NULL"},
                    {"name": "name", "type": "TEXT", "constraints": "NOT NULL"},
                    {"name": "email", "type": "TEXT", "constraints": "UNIQUE"},
                    {"name": "age", "type": "INTEGER"},
                    {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
                ],
                primary_key="id"
            )
            await self.database_service.create_table(create_users_request)
            
            # Create posts table
            create_posts_request = CreateTableRequest(
                table_name="posts",
                columns=[
                    {"name": "id", "type": "INTEGER", "constraints": "NOT NULL"},
                    {"name": "user_id", "type": "INTEGER", "constraints": "NOT NULL"},
                    {"name": "title", "type": "TEXT", "constraints": "NOT NULL"},
                    {"name": "content", "type": "TEXT"},
                    {"name": "created_at", "type": "TEXT", "constraints": "DEFAULT CURRENT_TIMESTAMP"}
                ],
                primary_key="id",
                foreign_keys=[
                    {"column": "user_id", "ref_table": "users", "ref_column": "id"}
                ]
            )
            await self.database_service.create_table(create_posts_request)
            
            # Insert sample data
            insert_users_request = InsertRequest(
                table_name="users",
                data=[
                    {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "age": 30},
                    {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "age": 25},
                    {"id": 3, "name": "Charlie Brown", "email": "charlie@example.com", "age": 35}
                ]
            )
            await self.database_service.insert_data(insert_users_request)
            
            insert_posts_request = InsertRequest(
                table_name="posts",
                data=[
                    {"id": 1, "user_id": 1, "title": "Hello World", "content": "This is my first post!"},
                    {"id": 2, "user_id": 2, "title": "Database Tips", "content": "Here are some useful database tips."},
                    {"id": 3, "user_id": 1, "title": "Python Programming", "content": "Python is a great language for data science."}
                ]
            )
            await self.database_service.insert_data(insert_posts_request)
            
            logger.info("| ✅ Sample tables and data created successfully")
            
        except Exception as e:
            logger.warning(f"| ⚠️ Failed to create sample tables: {e}")
    
    @ecp.action(name="execute_sql",
                type="Database",
                description="Execute a SQL query.")
    async def execute_sql(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        """Execute a SQL query.
        
        Args:
            query (str): SQL query string
            parameters (Optional[Dict[str, Any]]): Query parameters
            
        Returns:
            Query result message
        """
        request = QueryRequest(query=query, parameters=parameters)
        result = await self.database_service.execute_query(request)
        
        if result.success:
            if result.data:
                return f"Query executed successfully. Rows returned: {result.row_count}. Data: {result.data}"
            else:
                return f"Query executed successfully. Rows affected: {result.row_count}"
        else:
            return f"Query failed: {result.message}"
    
    @ecp.action(name="create_table",
                type="Database",
                description="Create a table.")
    async def create_table(self, 
                           table_name: str,
                           columns: List[Dict[str, Any]],
                           primary_key: Optional[str] = None,
                           foreign_keys: Optional[List[Dict[str, Any]]] = None) -> str:
        """Create a table.
        
        Args:
            table_name (str): Name of the table to create
            columns (List[Dict[str, Any]]): List of column definitions, each dict should have 'name', 'type', and optional 'constraints'
            primary_key (Optional[str]): Primary key column name
            foreign_keys (Optional[List[Dict[str, Any]]]): List of foreign key constraints, each dict should have 'column', 'ref_table', 'ref_column'
            
        Returns:
            str: Operation result message
        """
        request = CreateTableRequest(
            table_name=table_name,
            columns=columns,
            primary_key=primary_key,
            foreign_keys=foreign_keys
        )
        result = await self.database_service.create_table(request)
        
        if result.success:
            return f"Table '{table_name}' created successfully"
        else:
            return f"Failed to create table: {result.message}"
    
    @ecp.action(name="insert_data",
                type="Database",
                description="Insert data into a table.")
    async def insert_data(self, 
                          table_name: str,
                          data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> str:
        """Insert data into a table.
        
        Args:
            table_name (str): Name of the table to insert data into
            data (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to insert, can be a single record dict or list of record dicts
            
        Returns:
            str: Operation result message
        """
        request = InsertRequest(table_name=table_name, data=data)
        result = await self.database_service.insert_data(request)
        
        if result.success:
            return f"Data inserted successfully into '{table_name}'. Rows affected: {result.row_count}"
        else:
            return f"Failed to insert data: {result.message}"
    
    @ecp.action(name="select_data",
                type="Database",
                description="Select data from a table.")
    async def select_data(self, 
                          table_name: str,
                          columns: Optional[List[str]] = None,
                          where_clause: Optional[str] = None,
                          where_params: Optional[Dict[str, Any]] = None,
                          order_by: Optional[str] = None,
                          limit: Optional[int] = None) -> str:
        """Select data from a table.
        
        Args:
            table_name (str): Name of the table to select data from
            columns (Optional[List[str]]): List of column names to select, None for all columns
            where_clause (Optional[str]): WHERE clause for filtering data
            where_params (Optional[Dict[str, Any]]): Parameters for WHERE clause placeholders
            order_by (Optional[str]): ORDER BY clause for sorting results
            limit (Optional[int]): Maximum number of rows to return
            
        Returns:
            str: Query result message with data
        """
        request = SelectRequest(
            table_name=table_name,
            columns=columns,
            where_clause=where_clause,
            where_params=where_params,
            order_by=order_by,
            limit=limit
        )
        result = await self.database_service.select_data(request)
        
        if result.success:
            if result.data:
                return f"Query executed successfully. Rows returned: {result.row_count}. Data: {result.data}"
            else:
                return f"Query executed successfully. No data found."
        else:
            return f"Query failed: {result.message}"
    
    @ecp.action(name="update_data",
                type="Database",
                description="Update data in a table.")
    async def update_data(self, 
                          table_name: str,
                          data: Dict[str, Any],
                          where_clause: str,
                          where_params: Optional[Dict[str, Any]] = None) -> str:
        """Update data in a table.
        
        Args:
            table_name (str): Name of the table to update data in
            data (Dict[str, Any]): Dictionary of column names and new values to update
            where_clause (str): WHERE clause to identify which rows to update
            where_params (Optional[Dict[str, Any]]): Parameters for WHERE clause placeholders
            
        Returns:
            str: Operation result message
        """
        request = UpdateRequest(
            table_name=table_name,
            data=data,
            where_clause=where_clause,
            where_params=where_params
        )
        result = await self.database_service.update_data(request)
        
        if result.success:
            return f"Data updated successfully in '{table_name}'. Rows affected: {result.row_count}"
        else:
            return f"Failed to update data: {result.message}"
    
    @ecp.action(name="delete_data",
                type="Database",
                description="Delete data from a table.")
    async def delete_data(self, 
                          table_name: str,
                          where_clause: str,
                          where_params: Optional[Dict[str, Any]] = None) -> str:
        """Delete data from a table.
        
        Args:
            table_name (str): Name of the table to delete data from
            where_clause (str): WHERE clause to identify which rows to delete
            where_params (Optional[Dict[str, Any]]): Parameters for WHERE clause placeholders
            
        Returns:
            str: Operation result message
        """
        request = DeleteRequest(
            table_name=table_name,
            where_clause=where_clause,
            where_params=where_params
        )
        result = await self.database_service.delete_data(request)
        
        if result.success:
            return f"Data deleted successfully from '{table_name}'. Rows affected: {result.row_count}"
        else:
            return f"Failed to delete data: {result.message}"
    
    @ecp.action(name="get_tables",
                type="Database",
                description="Get information about all tables.")
    async def get_tables(self) -> str:
        """Get information about all tables.
        
        Returns:
            str: Table information message with table names, column counts, and row counts
        """
        request = GetTablesRequest()
        result = await self.database_service.get_tables(request)
        
        if result.success:
            if result.tables:
                table_info = []
                for table in result.tables:
                    table_info.append(f"Table: {table.name}, Columns: {len(table.columns)}, Rows: {table.row_count}")
                return f"Found {len(result.tables)} tables:\n" + "\n".join(table_info)
            else:
                return "No tables found in the database"
        else:
            return f"Failed to get tables: {result.message}"
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the database environment."""
        try:
            db_info = await self.database_service.get_database_info()
            return {
                "base_dir": str(self.base_dir),
                "database_path": str(self.base_dir / "database.db"),
                "is_connected": db_info.is_connected,
                "total_tables": db_info.total_tables,
                "tables": [
                    {
                        "name": table.name,
                        "columns": table.columns,
                        "row_count": table.row_count
                    }
                    for table in db_info.tables
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get database state: {e}")
            return {
                "base_dir": str(self.base_dir),
                "database_path": str(self.base_dir / "database.db"),
                "is_connected": False,
                "error": str(e)
            }