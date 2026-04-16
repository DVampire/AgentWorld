"""Database Environment for AgentWorld - provides database operations as an environment."""

import os
from typing import Any, Dict, List, Union, Optional, Type
from pydantic import BaseModel, Field, ConfigDict

from src.environment.database.service import DatabaseService
from src.environment.database.types import (
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
from src.environment.server import environment_manager
from src.environment.types import Environment
from src.utils import dedent
from src.registry import ENVIRONMENT

@ENVIRONMENT.register_module(force=True)
class DatabaseEnvironment(Environment):
    """Database Environment that provides database operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="database", description="The name of the database environment.")
    description: str = Field(default="Database environment for SQLite database operations", description="The description of the database environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the database environment including tables and data.",
        }
    }, description="The metadata of the database environment.")
    require_grad: bool = Field(default=False, description="Whether the environment requires gradients")
    
    def __init__(
        self,
        base_dir: str,
        auto_connect: bool = True,
        create_sample_tables: bool = True,
        require_grad: bool = False,
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
    
    @environment_manager.action(name="execute_sql",
                description="Execute a SQL query.")
    async def execute_sql(self, query: str, parameters: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Execute a SQL query.
        
        Args:
            query (str): SQL query string
            parameters (Optional[Dict[str, Any]]): Query parameters
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = QueryRequest(query=query, parameters=parameters)
            result = await self.database_service.execute_query(request)
            
            extra = result.extra.copy() if result.extra else {}
            
            if result.success:
                if "data" in extra:
                    message = f"Query executed successfully. Rows returned: {extra.get('row_count', 0)}"
                else:
                    message = f"Query executed successfully. Rows affected: {extra.get('row_count', 0)}"
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Query failed: {str(e)}",
                "extra": {"error": str(e), "query": query}
            }
    
    @environment_manager.action(name="create_table",
                description="Create a table.")
    async def create_table(self, 
                           table_name: str,
                           columns: List[Dict[str, Any]],
                           primary_key: Optional[str] = None,
                           foreign_keys: Optional[List[Dict[str, Any]]] = None,
                           **kwargs) -> Dict[str, Any]:
        """Create a table.
        
        Args:
            table_name (str): Name of the table to create
            columns (List[Dict[str, Any]]): List of column definitions, each dict should have 'name', 'type', and optional 'constraints'
            primary_key (Optional[str]): Primary key column name
            foreign_keys (Optional[List[Dict[str, Any]]]): List of foreign key constraints, each dict should have 'column', 'ref_table', 'ref_column'
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = CreateTableRequest(
                table_name=table_name,
                columns=columns,
                primary_key=primary_key,
                foreign_keys=foreign_keys
            )
            result = await self.database_service.create_table(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["table_name"] = table_name
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create table: {str(e)}",
                "extra": {"error": str(e), "table_name": table_name}
            }
    
    @environment_manager.action(name="insert_data",
                description="Insert data into a table.")
    async def insert_data(self, 
                          table_name: str,
                          data: Union[Dict[str, Any], List[Dict[str, Any]]],
                          **kwargs) -> Dict[str, Any]:
        """Insert data into a table.
        
        Args:
            table_name (str): Name of the table to insert data into
            data (Union[Dict[str, Any], List[Dict[str, Any]]]): Data to insert, can be a single record dict or list of record dicts
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = InsertRequest(table_name=table_name, data=data)
            result = await self.database_service.insert_data(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["table_name"] = table_name
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to insert data: {str(e)}",
                "extra": {"error": str(e), "table_name": table_name}
            }
    
    @environment_manager.action(name="select_data",
                description="Select data from a table.")
    async def select_data(self, 
                          table_name: str,
                          columns: Optional[List[str]] = None,
                          where_clause: Optional[str] = None,
                          where_params: Optional[Dict[str, Any]] = None,
                          order_by: Optional[str] = None,
                          limit: Optional[int] = None,
                          **kwargs) -> Dict[str, Any]:
        """Select data from a table.
        
        Args:
            table_name (str): Name of the table to select data from
            columns (Optional[List[str]]): List of column names to select, None for all columns
            where_clause (Optional[str]): WHERE clause for filtering data
            where_params (Optional[Dict[str, Any]]): Parameters for WHERE clause placeholders
            order_by (Optional[str]): ORDER BY clause for sorting results
            limit (Optional[int]): Maximum number of rows to return
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = SelectRequest(
                table_name=table_name,
                columns=columns,
                where_clause=where_clause,
                where_params=where_params,
                order_by=order_by,
                limit=limit
            )
            result = await self.database_service.select_data(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["table_name"] = table_name
            
            if result.success:
                if "data" in extra and extra["data"]:
                    message = f"Query executed successfully. Rows returned: {extra.get('row_count', 0)}"
                else:
                    message = f"Query executed successfully. No data found."
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Query failed: {str(e)}",
                "extra": {"error": str(e), "table_name": table_name}
            }
    
    @environment_manager.action(name="update_data",
                description="Update data in a table.")
    async def update_data(self, 
                          table_name: str,
                          data: Dict[str, Any],
                          where_clause: str,
                          where_params: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """Update data in a table.
        
        Args:
            table_name (str): Name of the table to update data in
            data (Dict[str, Any]): Dictionary of column names and new values to update
            where_clause (str): WHERE clause to identify which rows to update
            where_params (Optional[Dict[str, Any]]): Parameters for WHERE clause placeholders
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = UpdateRequest(
                table_name=table_name,
                data=data,
                where_clause=where_clause,
                where_params=where_params
            )
            result = await self.database_service.update_data(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["table_name"] = table_name
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to update data: {str(e)}",
                "extra": {"error": str(e), "table_name": table_name}
            }
    
    @environment_manager.action(name="delete_data",
                description="Delete data from a table.")
    async def delete_data(self, 
                          table_name: str,
                          where_clause: str,
                          where_params: Optional[Dict[str, Any]] = None,
                          **kwargs) -> Dict[str, Any]:
        """Delete data from a table.
        
        Args:
            table_name (str): Name of the table to delete data from
            where_clause (str): WHERE clause to identify which rows to delete
            where_params (Optional[Dict[str, Any]]): Parameters for WHERE clause placeholders
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = DeleteRequest(
                table_name=table_name,
                where_clause=where_clause,
                where_params=where_params
            )
            result = await self.database_service.delete_data(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["table_name"] = table_name
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete data: {str(e)}",
                "extra": {"error": str(e), "table_name": table_name}
            }
    
    @environment_manager.action(name="get_tables",
                description="Get information about all tables.")
    async def get_tables(self, **kwargs) -> Dict[str, Any]:
        """Get information about all tables.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = GetTablesRequest()
            result = await self.database_service.get_tables(request)
            
            extra = result.extra.copy() if result.extra else {}
            
            if result.success:
                if "tables" in extra and extra["tables"]:
                    table_info = []
                    for table in extra["tables"]:
                        table_info.append(f"Table: {table['name']}, Columns: {len(table['columns'])}, Rows: {table['row_count']}")
                    message = f"Found {len(extra['tables'])} tables:\n" + "\n".join(table_info)
                else:
                    message = "No tables found in the database"
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get tables: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_state(self, **kwargs) -> Dict[str, Any]:
        """Get the current state of the database environment."""
        try:
            db_info = await self.database_service.get_database_info()
            
            extra = db_info.extra
            state = dedent(f"""
                <info>
                Database File Path: {str(self.base_dir / "database.db")}
                Is Connected: {db_info.extra.get("is_connected", False)}
                Total Tables: {db_info.extra.get("total_tables", 0)}
                Tables: {", ".join([table["name"] for table in db_info.extra.get("tables", [])])}
                </info>
            """)
            return {
                "state": state,
                "extra": extra
            }   
        except Exception as e:
            logger.error(f"Failed to get database state: {e}")
            return {
                "state": f"Failed to get database state: {str(e)}",
                "extra": {"error": str(e)}
            }