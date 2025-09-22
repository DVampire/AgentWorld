"""FAISS-specific types and models."""

from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class FaissSearchRequest(BaseModel):
    """Request for FAISS similarity search."""
    query: str = Field(..., description="Query text to search for")
    k: int = Field(4, ge=1, le=1000, description="Number of documents to return")
    filter: Optional[Union[Dict[str, Any], Callable]] = Field(None, description="Filter by metadata")
    fetch_k: int = Field(20, ge=1, le=10000, description="Number of documents to fetch before filtering")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")


class FaissSearchResult(BaseModel):
    """Result of FAISS similarity search."""
    documents: List[Document] = Field(..., description="List of similar documents")
    scores: List[float] = Field(..., description="Similarity scores for each document")
    total_found: int = Field(..., description="Total number of documents found")


class FaissAddRequest(BaseModel):
    """Request for adding documents to FAISS."""
    texts: List[str] = Field(..., description="List of texts to add")
    metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Metadata for each text")
    ids: Optional[List[str]] = Field(None, description="Custom IDs for each text")


class FaissAddResult(BaseModel):
    """Result of adding documents to FAISS."""
    ids: List[str] = Field(..., description="IDs of added documents")
    count: int = Field(..., description="Number of documents added")


class FaissDeleteRequest(BaseModel):
    """Request for deleting documents from FAISS."""
    ids: List[str] = Field(..., description="IDs of documents to delete")


class FaissDeleteResult(BaseModel):
    """Result of deleting documents from FAISS."""
    deleted_count: int = Field(..., description="Number of documents deleted")
    success: bool = Field(..., description="Whether deletion was successful")


class FaissIndexInfo(BaseModel):
    """Information about the FAISS index."""
    total_documents: int = Field(..., description="Total number of documents in index")
    embedding_dimension: int = Field(..., description="Dimension of embeddings")
    index_type: str = Field(..., description="Type of FAISS index")
    distance_strategy: str = Field(..., description="Distance strategy used")


class FaissConfig(BaseModel):
    """Configuration for FAISS environment."""
    base_dir: str = Field(..., description="Base directory for FAISS storage")
    index_name: str = Field("index", description="Name of the FAISS index")
    embedding_model: str = Field("text-embedding-ada-002", description="Embedding model to use")
    distance_strategy: str = Field("cosine", description="Distance strategy (euclidean, cosine, max_inner_product)")
    normalize_L2: bool = Field(False, description="Whether to normalize L2")
    max_documents: int = Field(1000000, description="Maximum number of documents")
    auto_save: bool = Field(False, description="Whether to auto-save the index")
    save_interval: int = Field(100, description="Save interval in number of operations")
