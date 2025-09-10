from __future__ import annotations

from typing import Optional, Union
from pathlib import Path


class FileSystemError(Exception):
    """Base exception for the new async file system."""
    
    def __init__(self, message: str, path: Optional[Union[str, Path]] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.path = str(path) if path else None
        self.error_code = error_code
    
    def __str__(self) -> str:
        if self.path:
            return f"{self.message} (path: {self.path})"
        return self.message


class InvalidPathError(FileSystemError):
    """Raised when a provided path is invalid or malformed."""
    
    def __init__(self, message: str, path: Optional[Union[str, Path]] = None):
        super().__init__(message, path, "INVALID_PATH")


class PathTraversalError(FileSystemError):
    """Raised when a resolved path escapes the configured base directory sandbox."""
    
    def __init__(self, message: str, path: Optional[Union[str, Path]] = None, base_dir: Optional[Union[str, Path]] = None):
        super().__init__(message, path, "PATH_TRAVERSAL")
        self.base_dir = str(base_dir) if base_dir else None


class NotFoundError(FileSystemError):
    """Raised when a file or directory does not exist."""
    
    def __init__(self, message: str, path: Optional[Union[str, Path]] = None):
        super().__init__(message, path, "NOT_FOUND")


class ConflictError(FileSystemError):
    """Raised on conflicting operations (e.g., destination exists)."""
    
    def __init__(self, message: str, path: Optional[Union[str, Path]] = None, conflict_type: Optional[str] = None):
        super().__init__(message, path, "CONFLICT")
        self.conflict_type = conflict_type


class PermissionDeniedError(FileSystemError):
    """Raised when the operation is not permitted by the OS or policy."""
    
    def __init__(self, message: str, path: Optional[Union[str, Path]] = None, operation: Optional[str] = None):
        super().__init__(message, path, "PERMISSION_DENIED")
        self.operation = operation


class UnsupportedTypeError(FileSystemError):
    """Raised for unsupported file types or operations."""
    
    def __init__(self, message: str, file_type: Optional[str] = None, path: Optional[Union[str, Path]] = None):
        super().__init__(message, path, "UNSUPPORTED_TYPE")
        self.file_type = file_type


class InvalidArgumentError(FileSystemError):
    """Raised when arguments are missing or invalid for an operation."""
    
    def __init__(self, message: str, argument: Optional[str] = None, value: Optional[str] = None):
        super().__init__(message, error_code="INVALID_ARGUMENT")
        self.argument = argument
        self.value = value


class CacheError(FileSystemError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(message, error_code="CACHE_ERROR")
        self.operation = operation


class StorageError(FileSystemError):
    """Raised when storage backend operations fail."""
    
    def __init__(self, message: str, path: Optional[Union[str, Path]] = None, operation: Optional[str] = None):
        super().__init__(message, path, "STORAGE_ERROR")
        self.operation = operation


class HandlerError(FileSystemError):
    """Raised when content handler operations fail."""
    
    def __init__(self, message: str, handler_type: Optional[str] = None, path: Optional[Union[str, Path]] = None):
        super().__init__(message, path, "HANDLER_ERROR")
        self.handler_type = handler_type


class LockError(FileSystemError):
    """Raised when file locking operations fail."""
    
    def __init__(self, message: str, path: Optional[Union[str, Path]] = None, operation: Optional[str] = None):
        super().__init__(message, path, "LOCK_ERROR")
        self.operation = operation

