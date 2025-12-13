"""FAISS-specific exceptions."""

from typing import Any, Dict, Optional


class FaissError(Exception):
    """Base exception for FAISS operations."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class FaissIndexError(FaissError):
    """Error related to FAISS index operations."""
    pass


class FaissEmbeddingError(FaissError):
    """Error related to embedding operations."""
    pass


class FaissDocumentError(FaissError):
    """Error related to document operations."""
    pass


class FaissSearchError(FaissError):
    """Error related to search operations."""
    pass


class FaissStorageError(FaissError):
    """Error related to storage operations."""
    pass


class FaissConfigurationError(FaissError):
    """Error related to configuration issues."""
    pass
