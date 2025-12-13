"""Database environment exceptions."""

from typing import Optional


class DatabaseError(Exception):
    """Base exception for database operations."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidQueryError(DatabaseError):
    """Raised when a SQL query is invalid."""
    pass


class TableNotFoundError(DatabaseError):
    """Raised when a table is not found."""
    pass


class ColumnNotFoundError(DatabaseError):
    """Raised when a column is not found."""
    pass


class ConstraintViolationError(DatabaseError):
    """Raised when a database constraint is violated."""
    pass


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class TransactionError(DatabaseError):
    """Raised when a database transaction fails."""
    pass
