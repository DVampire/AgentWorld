"""FAISS Vector Store Service for AgentWorld."""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os

import numpy as np
import uuid
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy

from src.logger import logger
from src.environments.faiss.exceptions import (
    FaissIndexError, 
    FaissDocumentError, 
    FaissSearchError, 
    FaissStorageError,
    FaissConfigurationError
)
from src.environments.faiss.types import (
    FaissSearchRequest, 
    FaissSearchResult, 
    FaissAddRequest,
    FaissAddResult,
    FaissDeleteRequest,
    FaissDeleteResult,
    FaissIndexInfo, 
    FaissConfig
)
from src.environments.faiss.faiss import FAISS, dependable_faiss_import


class FaissService:
    """Async FAISS vector store service with embedding support."""
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        embedding_function: Optional[Embeddings] = None,
        config: Optional[FaissConfig] = None
    ):
        """Initialize the FAISS service.
        
        Args:
            base_dir: Base directory for FAISS storage
            embedding_function: Embedding function to use
            config: Configuration for the service
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or FaissConfig(base_dir=str(base_dir))
        self.embedding_function = embedding_function
        self.vector_store: Optional[FAISS] = None
        self._operation_count = 0
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _initialize_index(self) -> None:
        """Initialize the FAISS index."""
        try:
            self.index_path = self.base_dir / f"{self.config.index_name}.faiss"
            self.pkl_path = self.base_dir / f"{self.config.index_name}.pkl"
            
            if self.index_path.exists() and self.pkl_path.exists():
                # Load existing index
                self._load_index()
            else:
                # Create new index
                self._create_index()
                
        except Exception as e:
            raise FaissIndexError(f"Failed to initialize FAISS index: {e}")
    
    def _create_index(self) -> None:
        """Create a new FAISS index."""
        if not self.embedding_function:
            raise FaissConfigurationError("Embedding function is required to create index")
        
        try:
            # Get embedding dimension
            test_embedding = self.embedding_function.embed_query("test")
            dimension = len(test_embedding)
            
            # Create FAISS index
            faiss = dependable_faiss_import()
            distance_strategy = self._get_distance_strategy()
            
            if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
                index = faiss.IndexFlatIP(dimension)
            else:
                index = faiss.IndexFlatL2(dimension)
            
            # Create vector store
            self.vector_store = FAISS(
                embedding_function=self.embedding_function,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                distance_strategy=distance_strategy,
                normalize_L2=self.config.normalize_L2
            )
            
            logger.info(f"| 🔍 Created new FAISS index with dimension {dimension}")
            
        except Exception as e:
            raise FaissIndexError(f"Failed to create FAISS index: {e}")
    
    def _load_index(self) -> None:
        """Load existing FAISS index from disk."""
        try:
            if not self.embedding_function:
                raise FaissConfigurationError("Embedding function is required to load index")
            
            self.vector_store = FAISS.load_local(
                folder_path=str(self.base_dir),
                embeddings=self.embedding_function,
                index_name=self.config.index_name,
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"| 🔍 Loaded existing FAISS index from {self.base_dir}")
            
        except Exception as e:
            raise FaissIndexError(f"Failed to load FAISS index: {e}")
    
    def _get_distance_strategy(self) -> DistanceStrategy:
        """Get distance strategy from config."""
        strategy_map = {
            "euclidean": DistanceStrategy.EUCLIDEAN_DISTANCE,
            "cosine": DistanceStrategy.COSINE,
            "max_inner_product": DistanceStrategy.MAX_INNER_PRODUCT
        }
        return strategy_map.get(self.config.distance_strategy, DistanceStrategy.EUCLIDEAN_DISTANCE)
    
    async def add_documents(self, request: FaissAddRequest) -> FaissAddResult:
        """Add documents to the FAISS index.
        
        Args:
            request: Add request with texts and metadata
            
        Returns:
            Add result with IDs and count
        """
        if not self.vector_store:
            raise FaissIndexError("FAISS index not initialized")
        
        try:
            # Filter out empty texts
            valid_texts = []
            valid_metadatas = []
            for i, text in enumerate(request.texts):
                if text and text.strip():  # Skip empty or whitespace-only texts
                    valid_texts.append(text)
                    if request.metadatas and i < len(request.metadatas):
                        valid_metadatas.append(request.metadatas[i])
                    else:
                        valid_metadatas.append({})
            
            if not valid_texts:
                logger.info("| ⚠️ No valid texts to add (all texts were empty)")
                return FaissAddResult(ids=[], count=0)

            ids = []
            documents = []
            for text, metadata in zip(valid_texts, valid_metadatas):
                ids.append(str(uuid.uuid4()))
                documents.append(Document(page_content=text, metadata=metadata or {}))
            
            # Add documents
            ids = await self.vector_store.aadd_documents(
                documents=documents,
                ids=ids
            )
            
            self._operation_count += 1
            await self._auto_save()
            
            logger.info(f"| ➕ Added {len(ids)} documents to FAISS index")
            return FaissAddResult(ids=ids, count=len(ids))
            
        except Exception as e:
            raise FaissDocumentError(f"Failed to add documents: {e}")
    
    async def search_similar(self, request: FaissSearchRequest) -> FaissSearchResult:
        """Search for similar documents.
        
        Args:
            request: Search request with query and parameters
            
        Returns:
            Search result with documents and scores
        """
        if not self.vector_store:
            raise FaissIndexError("FAISS index not initialized")
        
        try:
            # Perform similarity search
            docs_and_scores = await self.vector_store.asimilarity_search_with_score(
                query=request.query,
                k=request.k,
                filter=request.filter,
                fetch_k=request.fetch_k
            )
            
            # Apply score threshold if specified
            if request.score_threshold is not None:
                docs_and_scores = [
                    (doc, score) for doc, score in docs_and_scores 
                    if score >= request.score_threshold
                ]
            
            documents = [doc for doc, _ in docs_and_scores]
            scores = [score for _, score in docs_and_scores]
            
            logger.info(f"| 🔍 Found {len(documents)} similar documents for query: {request.query[:50]}...")
            return FaissSearchResult(
                documents=documents,
                scores=scores,
                total_found=len(documents)
            )
            
        except Exception as e:
            raise FaissSearchError(f"Failed to search documents: {e}")
    
    async def delete_documents(self, request: FaissDeleteRequest) -> FaissDeleteResult:
        """Delete documents from the FAISS index.
        
        Args:
            request: Delete request with document IDs
            
        Returns:
            Delete result with count and success status
        """
        if not self.vector_store:
            raise FaissIndexError("FAISS index not initialized")
        
        try:
            # Delete documents
            result = await self.vector_store.adelete(ids=request.ids)
            
            self._operation_count += 1
            await self._auto_save()
            
            deleted_count = len(request.ids) if result else 0
            logger.info(f"| 🗑️ Deleted {deleted_count} documents from FAISS index")
            return FaissDeleteResult(
                deleted_count=deleted_count,
                success=result is not False
            )
            
        except ValueError as e:
            # Handle case where some IDs don't exist
            if "Some specified ids do not exist" in str(e):
                logger.warning(f"| ⚠️ Some IDs not found during deletion: {e}")
                return FaissDeleteResult(
                    deleted_count=0,
                    success=True  # Still consider it successful, just no documents deleted
                )
            else:
                raise FaissDocumentError(f"Failed to delete documents: {e}")
        except Exception as e:
            raise FaissDocumentError(f"Failed to delete documents: {e}")
    
    async def get_index_info(self) -> FaissIndexInfo:
        """Get information about the FAISS index.
        
        Returns:
            Index information
        """
        if not self.vector_store:
            raise FaissIndexError("FAISS index not initialized")
        
        try:
            total_documents = len(self.vector_store.index_to_docstore_id)
            
            # Get embedding dimension
            if hasattr(self.vector_store.index, 'd'):
                embedding_dimension = self.vector_store.index.d
            else:
                embedding_dimension = 0
            
            return FaissIndexInfo(
                total_documents=total_documents,
                embedding_dimension=embedding_dimension,
                index_type=type(self.vector_store.index).__name__,
                distance_strategy=self.config.distance_strategy
            )
            
        except Exception as e:
            raise FaissIndexError(f"Failed to get index info: {e}")
    
    async def save_index(self) -> None:
        """Save the FAISS index to disk."""
        if not self.vector_store:
            raise FaissIndexError("FAISS index not initialized")
        
        try:
            self.vector_store.save_local(
                folder_path=str(self.base_dir),
                index_name=self.config.index_name
            )
            logger.info(f"| 💾 Saved FAISS index to {self.base_dir}")
            
        except Exception as e:
            raise FaissStorageError(f"Failed to save FAISS index: {e}")
    
    async def _auto_save(self) -> None:
        """Auto-save the index if configured."""
        if self.config.auto_save and self._operation_count % self.config.save_interval == 0:
            await self.save_index()
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Delete the index from disk
            if self.index_path.exists():
                os.remove(self.index_path)
            if self.pkl_path.exists():
                os.remove(self.pkl_path)
        except Exception as e:
            logger.warning(f"| ⚠️ Error during FAISS cleanup: {e}")
