"""FAISS Vector Store Service for AgentWorld - Native implementation without langchain."""

import asyncio
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import os
import uuid

import numpy as np

from src.logger import logger
from src.environment.protocol.types import ActionResult
from src.environment.faiss.exceptions import (
    FaissIndexError, 
    FaissDocumentError, 
    FaissSearchError, 
    FaissStorageError,
    FaissConfigurationError,
    FaissEmbeddingError
)
from src.environment.faiss.types import (
    FaissSearchRequest, 
    FaissAddRequest,
    FaissDeleteRequest,
    FaissIndexInfo, 
    FaissConfig
)
from src.model import model_manager
from src.message import HumanMessage


def dependable_faiss_import():
    """Import faiss if available, otherwise raise error."""
    try:
        import faiss
        return faiss
    except ImportError:
        raise ImportError(
            "Could not import faiss python package. "
            "Please install it with `pip install faiss-gpu` (for CUDA supported GPU) "
            "or `pip install faiss-cpu`."
        )


class Document:
    """Simple document class to replace langchain Document."""
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class FaissService:
    """Async FAISS vector store service with model_manager embedding support."""
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        model_name: Optional[str] = None,
        config: Optional[FaissConfig] = None
    ):
        """Initialize the FAISS service.
        
        Args:
            base_dir: Base directory for FAISS storage
            model_name: Model name for embeddings (defaults to config value)
            config: Configuration for the service
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or FaissConfig(base_dir=str(base_dir))
        self.model_name = model_name or self.config.model_name
        
        # FAISS components
        self.faiss = dependable_faiss_import()
        self.index: Optional[Any] = None
        self.docstore: Dict[str, Document] = {}  # id -> Document
        self.id_to_index: Dict[str, int] = {}  # document id -> FAISS index position
        self.index_to_id: Dict[int, str] = {}  # FAISS index position -> document id
        
        self._operation_count = 0
        self._embedding_dimension: Optional[int] = None
        self._index_creation_lock = asyncio.Lock()  # Lock to prevent concurrent index creation
        
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
    
    async def _get_embedding_dimension(self) -> int:
        """Get embedding dimension by making a test call."""
        if self._embedding_dimension is not None:
            return self._embedding_dimension
        
        try:
            test_message = [HumanMessage(content="test")]
            response = await model_manager.aembedding(
                model=self.model_name,
                messages=test_message
            )
            
            if not response.success:
                raise FaissConfigurationError(f"Failed to get embedding dimension: {response.message}")
            
            embedding = response.extra["embeddings"]
            
            # Handle single vs multiple embeddings
            # Single message: returns single np.array with shape (dim,)
            # Multiple messages: returns list of np.arrays, each with shape (dim,)
            if isinstance(embedding, np.ndarray):
                # Single embedding: shape is (dim,)
                if embedding.ndim == 1:
                    self._embedding_dimension = embedding.shape[0]
                else:
                    self._embedding_dimension = embedding.shape[-1]
            elif isinstance(embedding, list) and len(embedding) > 0:
                # Multiple embeddings: list of np.arrays
                first_emb = embedding[0]
                if isinstance(first_emb, np.ndarray):
                    if first_emb.ndim == 1:
                        self._embedding_dimension = first_emb.shape[0]
                    else:
                        self._embedding_dimension = first_emb.shape[-1]
                else:
                    # Fallback: assume it's a list of numbers
                    self._embedding_dimension = len(first_emb)
            else:
                raise FaissConfigurationError(f"Invalid embedding response format: {type(embedding)}")
            
            return self._embedding_dimension
            
        except Exception as e:
            raise FaissConfigurationError(f"Failed to get embedding dimension: {e}")
    
    def _create_index(self) -> None:
        """Create a new FAISS index."""
        try:
            # Get embedding dimension synchronously (will be set when first embedding is made)
            # For now, we'll create index lazily when first document is added
            logger.info(f"| 🔍 FAISS index will be created on first document addition")
            
        except Exception as e:
            raise FaissIndexError(f"Failed to create FAISS index: {e}")
    
    async def _ensure_index_created(self) -> None:
        """Ensure FAISS index is created with correct dimension."""
        # Double-check pattern with lock to prevent concurrent creation
        if self.index is not None:
            return
        
        async with self._index_creation_lock:
            # Check again after acquiring lock
            if self.index is not None:
                return
            
            dimension = await self._get_embedding_dimension()
            
            # Create FAISS index based on distance strategy
            if self.config.distance_strategy == "max_inner_product":
                self.index = self.faiss.IndexFlatIP(dimension)
            elif self.config.distance_strategy == "cosine":
                # For cosine similarity, use L2 index with normalization
                self.index = self.faiss.IndexFlatL2(dimension)
            else:  # euclidean
                self.index = self.faiss.IndexFlatL2(dimension)
            
            logger.info(f"| 🔍 Created FAISS index with dimension {dimension}, strategy: {self.config.distance_strategy}")
    
    def _load_index(self) -> None:
        """Load existing FAISS index from disk."""
        try:
            # Load FAISS index
            self.index = self.faiss.read_index(str(self.index_path))
            
            # Load docstore and mappings
            with open(self.pkl_path, 'rb') as f:
                data = pickle.load(f)
                self.docstore = data.get("docstore", {})
                self.id_to_index = data.get("id_to_index", {})
                self.index_to_id = data.get("index_to_id", {})
                self._embedding_dimension = data.get("embedding_dimension")
            
            logger.info(f"| 🔍 Loaded existing FAISS index from {self.base_dir} with {len(self.docstore)} documents")
            
        except Exception as e:
            raise FaissIndexError(f"Failed to load FAISS index: {e}")
    
    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts using model_manager.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            np.ndarray with shape (num_texts, dim)
        """
        try:
            messages = [HumanMessage(content=text) for text in texts]
            response = await model_manager.aembedding(
                model=self.model_name,
                messages=messages
            )
            
            if not response.success:
                raise FaissEmbeddingError(f"Failed to get embeddings: {response.message}")
            
            embeddings = response.extra["embeddings"]
            
            # Handle single vs multiple embeddings
            # Single message: returns single np.array with shape (dim,)
            # Multiple messages: returns list of np.arrays, each with shape (dim,)
            if isinstance(embeddings, np.ndarray):
                # Single embedding: shape is (dim,), reshape to (1, dim)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)
                return embeddings.astype(np.float32)
            elif isinstance(embeddings, list):
                # Multiple embeddings: list of np.arrays, each with shape (dim,)
                # Stack them into (num_texts, dim)
                embeddings_list = []
                for emb in embeddings:
                    if isinstance(emb, np.ndarray):
                        if emb.ndim == 1:
                            embeddings_list.append(emb)
                        else:
                            embeddings_list.append(emb.flatten())
                    else:
                        # Fallback: convert to array
                        embeddings_list.append(np.array(emb))
                
                if embeddings_list:
                    embeddings_array = np.stack(embeddings_list, axis=0).astype(np.float32)
                    return embeddings_array
                else:
                    raise FaissEmbeddingError("Empty embeddings list")
            else:
                raise FaissEmbeddingError(f"Unexpected embedding format: {type(embeddings)}")
                
        except Exception as e:
            raise FaissEmbeddingError(f"Failed to get embeddings: {e}")
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length for cosine similarity."""
        if self.config.distance_strategy == "cosine" or self.config.normalize_L2:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            return vectors / norms
        return vectors
    
    async def add_documents(self, request: FaissAddRequest) -> ActionResult:
        """Add documents to the FAISS index.
        
        Args:
            request: Add request with texts and metadata
            
        Returns:
            Action result with IDs and count in extra
        """
        try:
            # Ensure index is created
            await self._ensure_index_created()
            
            # Filter out empty texts
            valid_texts = []
            valid_metadatas = []
            valid_indices = []
            for i, text in enumerate(request.texts):
                if text and text.strip():
                    valid_texts.append(text)
                    if request.metadatas and i < len(request.metadatas):
                        valid_metadatas.append(request.metadatas[i])
                    else:
                        valid_metadatas.append({})
                    valid_indices.append(i)
            
            if not valid_texts:
                logger.info("| ⚠️ No valid texts to add (all texts were empty)")
                return ActionResult(
                    success=True,
                    message="No valid texts to add (all texts were empty)",
                    extra={"ids": [], "count": 0, "total_input": len(request.texts)}
                )
            
            # Generate IDs
            ids = []
            if request.ids:
                # Use provided IDs, but only for valid texts
                for idx in valid_indices:
                    if idx < len(request.ids):
                        ids.append(request.ids[idx])
                    else:
                        ids.append(str(uuid.uuid4()))
            else:
                ids = [str(uuid.uuid4()) for _ in valid_texts]
            
            # Get embeddings
            embeddings = await self._get_embeddings(valid_texts)
            embeddings = self._normalize_vectors(embeddings)
            
            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(embeddings)
            
            # Store documents and mappings
            for i, (text, metadata, doc_id) in enumerate(zip(valid_texts, valid_metadatas, ids)):
                idx = start_idx + i
                self.docstore[doc_id] = Document(page_content=text, metadata=metadata)
                self.id_to_index[doc_id] = idx
                self.index_to_id[idx] = doc_id
            
            self._operation_count += 1
            await self._auto_save()
            
            logger.info(f"| ➕ Added {len(ids)} documents to FAISS index")
            return ActionResult(
                success=True,
                message=f"Added {len(ids)} documents to FAISS index",
                extra={
                    "ids": ids,
                    "count": len(ids),
                    "total_input": len(request.texts),
                    "valid_input": len(valid_texts)
                }
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to add documents: {str(e)}",
                extra={"error": str(e)}
            )
    
    async def search_similar(self, request: FaissSearchRequest) -> ActionResult:
        """Search for similar documents.
        
        Args:
            request: Search request with query and parameters
            
        Returns:
            Action result with documents and scores in extra
        """
        if self.index is None or self.index.ntotal == 0:
            return ActionResult(
                success=False,
                message="FAISS index not initialized or empty",
                extra={"error": "FAISS index not initialized or empty"}
            )
        
        try:
            # Get query embedding
            query_embeddings = await self._get_embeddings([request.query])
            query_embedding = self._normalize_vectors(query_embeddings)[0:1]  # Keep 2D shape
            
            # Search
            k = min(request.k, self.index.ntotal)
            fetch_k = min(request.fetch_k or k, self.index.ntotal)
            
            distances, indices = self.index.search(query_embedding, fetch_k)
            
            # Convert distances to scores based on distance strategy
            if self.config.distance_strategy == "cosine":
                # Cosine similarity: 1 - distance (since we normalized, distance is 1 - cosine)
                scores = 1 - distances[0]
            elif self.config.distance_strategy == "max_inner_product":
                # Inner product is already the score
                scores = distances[0]
            else:  # euclidean
                # Convert L2 distance to similarity score (lower distance = higher similarity)
                # Use negative distance or inverse
                max_dist = np.max(distances[0]) if len(distances[0]) > 0 else 1.0
                scores = 1 - (distances[0] / (max_dist + 1e-8))
            
            # Get documents
            docs_and_scores = []
            for idx, score in zip(indices[0], scores):
                if idx < 0:  # FAISS returns -1 for invalid indices
                    continue
                doc_id = self.index_to_id.get(idx)
                if doc_id and doc_id in self.docstore:
                    doc = self.docstore[doc_id]
                    # Apply filter if specified
                    if request.filter:
                        if isinstance(request.filter, dict):
                            # Check if metadata matches filter
                            if not all(doc.metadata.get(k) == v for k, v in request.filter.items()):
                                continue
                        elif callable(request.filter):
                            # Callable filter
                            if not request.filter(doc.metadata):
                                continue
                    docs_and_scores.append((doc, float(score)))
            
            # Sort by score (descending)
            docs_and_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Apply score threshold
            if request.score_threshold is not None:
                docs_and_scores = [
                    (doc, score) for doc, score in docs_and_scores 
                    if score >= request.score_threshold
                ]
            
            # Take top k
            docs_and_scores = docs_and_scores[:k]
            
            documents = [doc for doc, _ in docs_and_scores]
            scores_list = [score for _, score in docs_and_scores]
            
            # Convert documents to dict for serialization
            documents_dict = [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ]
            
            logger.info(f"| 🔍 Found {len(documents)} similar documents for query: {request.query[:50]}...")
            return ActionResult(
                success=True,
                message=f"Found {len(documents)} similar documents",
                extra={
                    "documents": documents_dict,
                    "scores": scores_list,
                    "total_found": len(documents),
                    "query": request.query,
                    "k": request.k
                }
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to search documents: {str(e)}",
                extra={"error": str(e), "query": request.query}
            )
    
    async def delete_documents(self, request: FaissDeleteRequest) -> ActionResult:
        """Delete documents from the FAISS index.
        
        Note: FAISS doesn't support direct deletion. We mark documents as deleted
        by removing them from docstore and mappings, but the index remains unchanged.
        For true deletion, we would need to rebuild the index.
        
        Args:
            request: Delete request with document IDs
            
        Returns:
            Action result with count and success status in extra
        """
        if self.index is None:
            return ActionResult(
                success=False,
                message="FAISS index not initialized",
                extra={"error": "FAISS index not initialized"}
            )
        
        try:
            deleted_count = 0
            for doc_id in request.ids:
                if doc_id in self.docstore:
                    # Remove from docstore and mappings
                    idx = self.id_to_index.pop(doc_id, None)
                    if idx is not None:
                        self.index_to_id.pop(idx, None)
                    del self.docstore[doc_id]
                    deleted_count += 1
            
            self._operation_count += 1
            await self._auto_save()
            
            logger.info(f"| 🗑️ Deleted {deleted_count} documents from FAISS index")
            return ActionResult(
                success=True,
                message=f"Deleted {deleted_count} documents from FAISS index",
                extra={
                    "deleted_count": deleted_count,
                    "requested_ids": request.ids,
                    "total_requested": len(request.ids)
                }
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to delete documents: {str(e)}",
                extra={"error": str(e), "requested_ids": request.ids}
            )
    
    async def get_index_info(self) -> ActionResult:
        """Get information about the FAISS index.
        
        Returns:
            Action result with index information in extra
        """
        if self.index is None:
            return ActionResult(
                success=False,
                message="FAISS index not initialized",
                extra={"error": "FAISS index not initialized"}
            )
        
        try:
            total_documents = len(self.docstore)
            embedding_dimension = self.index.d if hasattr(self.index, 'd') else self._embedding_dimension or 0
            
            index_info = FaissIndexInfo(
                total_documents=total_documents,
                embedding_dimension=embedding_dimension,
                index_type=type(self.index).__name__,
                distance_strategy=self.config.distance_strategy
            )
            
            return ActionResult(
                success=True,
                message=f"FAISS index information retrieved successfully",
                extra={
                    "index_info": index_info.model_dump(),
                    "total_documents": total_documents,
                    "embedding_dimension": embedding_dimension,
                    "index_type": type(self.index).__name__,
                    "distance_strategy": self.config.distance_strategy,
                    "index_name": self.config.index_name
                }
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Failed to get index info: {str(e)}",
                extra={"error": str(e)}
            )
    
    async def save_index(self) -> None:
        """Save the FAISS index to disk."""
        if self.index is None:
            raise FaissIndexError("FAISS index not initialized")
        
        try:
            # Save FAISS index
            self.faiss.write_index(self.index, str(self.index_path))
            
            # Save docstore and mappings
            data = {
                "docstore": self.docstore,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
                "embedding_dimension": self._embedding_dimension
            }
            with open(self.pkl_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"| 💾 Saved FAISS index to {self.base_dir}")
            
        except Exception as e:
            raise FaissStorageError(f"Failed to save FAISS index: {e}")
    
    async def _auto_save(self) -> None:
        """Auto-save the index if configured."""
        if self.config.auto_save and self._operation_count % self.config.save_interval == 0:
            await self.save_index()
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Save before cleanup if needed
            if self.index is not None:
                await self.save_index()
        except Exception as e:
            logger.warning(f"| ⚠️ Error during FAISS cleanup: {e}")
