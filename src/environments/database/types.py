"""Database environment types."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request for executing a SQL query."""
    query: str = Field(description="SQL query to execute")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Query parameters")


class TableInfo(BaseModel):
    """Information about a database table."""
    name: str = Field(description="Table name")
    columns: List[Dict[str, Any]] = Field(description="Table columns")
    row_count: int = Field(description="Number of rows in the table")


class QueryResult(BaseModel):
    """Result of a SQL query execution."""
    success: bool = Field(description="Whether the query was successful")
    data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Query result data")
    row_count: int = Field(default=0, description="Number of rows returned")
    message: Optional[str] = Field(default=None, description="Success or error message")
    execution_time: Optional[float] = Field(default=None, description="Query execution time in seconds")


class DatabaseInfo(BaseModel):
    """Information about the database."""
    path: str = Field(description="Database file path")
    tables: List[TableInfo] = Field(description="List of tables in the database")
    total_tables: int = Field(description="Total number of tables")
    is_connected: bool = Field(description="Whether the database is connected")


class CreateTableRequest(BaseModel):
    """Request for creating a table."""
    table_name: str = Field(description="Name of the table to create")
    columns: List[Dict[str, Any]] = Field(description="Table column definitions")
    primary_key: Optional[str] = Field(default=None, description="Primary key column name")
    foreign_keys: Optional[List[Dict[str, Any]]] = Field(default=None, description="Foreign key constraints")


class InsertRequest(BaseModel):
    """Request for inserting data into a table."""
    table_name: str = Field(description="Name of the table")
    data: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(description="Data to insert")


class UpdateRequest(BaseModel):
    """Request for updating data in a table."""
    table_name: str = Field(description="Name of the table")
    data: Dict[str, Any] = Field(description="Data to update")
    where_clause: str = Field(description="WHERE clause for the update")
    where_params: Optional[Dict[str, Any]] = Field(default=None, description="Parameters for WHERE clause")


class DeleteRequest(BaseModel):
    """Request for deleting data from a table."""
    table_name: str = Field(description="Name of the table")
    where_clause: str = Field(description="WHERE clause for the delete")
    where_params: Optional[Dict[str, Any]] = Field(default=None, description="Parameters for WHERE clause")


class CreateTableResult(BaseModel):
    """Result of creating a table."""
    success: bool = Field(description="Whether the operation was successful")
    message: Optional[str] = Field(default=None, description="Success or error message")
    execution_time: Optional[float] = Field(default=None, description="Operation execution time in seconds")


class InsertResult(BaseModel):
    """Result of inserting data."""
    success: bool = Field(description="Whether the operation was successful")
    row_count: int = Field(default=0, description="Number of rows affected")
    message: Optional[str] = Field(default=None, description="Success or error message")
    execution_time: Optional[float] = Field(default=None, description="Operation execution time in seconds")


class UpdateResult(BaseModel):
    """Result of updating data."""
    success: bool = Field(description="Whether the operation was successful")
    row_count: int = Field(default=0, description="Number of rows affected")
    message: Optional[str] = Field(default=None, description="Success or error message")
    execution_time: Optional[float] = Field(default=None, description="Operation execution time in seconds")


class DeleteResult(BaseModel):
    """Result of deleting data."""
    success: bool = Field(description="Whether the operation was successful")
    row_count: int = Field(default=0, description="Number of rows affected")
    message: Optional[str] = Field(default=None, description="Success or error message")
    execution_time: Optional[float] = Field(default=None, description="Operation execution time in seconds")


class SelectRequest(BaseModel):
    """Request for selecting data from a table."""
    table_name: str = Field(description="Name of the table")
    columns: Optional[List[str]] = Field(default=None, description="List of columns to select")
    where_clause: Optional[str] = Field(default=None, description="WHERE clause")
    where_params: Optional[Dict[str, Any]] = Field(default=None, description="Parameters for WHERE clause")
    order_by: Optional[str] = Field(default=None, description="ORDER BY clause")
    limit: Optional[int] = Field(default=None, description="LIMIT clause")


class SelectResult(BaseModel):
    """Result of selecting data."""
    success: bool = Field(description="Whether the operation was successful")
    data: Optional[List[Dict[str, Any]]] = Field(default=None, description="Selected data")
    row_count: int = Field(default=0, description="Number of rows returned")
    message: Optional[str] = Field(default=None, description="Success or error message")
    execution_time: Optional[float] = Field(default=None, description="Operation execution time in seconds")


class GetTablesRequest(BaseModel):
    """Request for getting table information."""
    pass


class GetTablesResult(BaseModel):
    """Result of getting table information."""
    success: bool = Field(description="Whether the operation was successful")
    tables: List[TableInfo] = Field(default_factory=list, description="List of table information")
    message: Optional[str] = Field(default=None, description="Success or error message")
