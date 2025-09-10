"""Exception definitions for GitHub service."""

from __future__ import annotations

from typing import Optional, Union
from datetime import datetime


class GitHubError(Exception):
    """Base exception for GitHub service."""
    
    def __init__(
        self, 
        message: str, 
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        rate_limit_reset: Optional[datetime] = None
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.rate_limit_reset = rate_limit_reset
    
    def __str__(self) -> str:
        if self.status_code:
            return f"{self.message} (status: {self.status_code})"
        return self.message


class AuthenticationError(GitHubError):
    """Raised when GitHub authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", status_code: Optional[int] = None):
        super().__init__(message, status_code, "AUTHENTICATION_ERROR")


class RateLimitError(GitHubError):
    """Raised when GitHub API rate limit is exceeded."""
    
    def __init__(
        self, 
        message: str = "Rate limit exceeded", 
        status_code: Optional[int] = None,
        rate_limit_reset: Optional[datetime] = None
    ):
        super().__init__(message, status_code, "RATE_LIMIT_ERROR", rate_limit_reset)


class NotFoundError(GitHubError):
    """Raised when a GitHub resource is not found."""
    
    def __init__(self, message: str = "Resource not found", status_code: Optional[int] = None):
        super().__init__(message, status_code, "NOT_FOUND")


class ConflictError(GitHubError):
    """Raised when there's a conflict with GitHub operations."""
    
    def __init__(self, message: str = "Operation conflict", status_code: Optional[int] = None):
        super().__init__(message, status_code, "CONFLICT")


class ValidationError(GitHubError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str = "Validation failed", status_code: Optional[int] = None):
        super().__init__(message, status_code, "VALIDATION_ERROR")


class APIError(GitHubError):
    """Raised for general GitHub API errors."""
    
    def __init__(self, message: str = "API error", status_code: Optional[int] = None):
        super().__init__(message, status_code, "API_ERROR")


class NetworkError(GitHubError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str = "Network error", status_code: Optional[int] = None):
        super().__init__(message, status_code, "NETWORK_ERROR")


class TimeoutError(GitHubError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str = "Operation timeout", status_code: Optional[int] = None):
        super().__init__(message, status_code, "TIMEOUT_ERROR")


class PermissionError(GitHubError):
    """Raised when insufficient permissions for operation."""
    
    def __init__(self, message: str = "Insufficient permissions", status_code: Optional[int] = None):
        super().__init__(message, status_code, "PERMISSION_ERROR")
