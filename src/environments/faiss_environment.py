"""FAISS Vector Store Environment for AgentWorld."""

from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from src.environments.faiss.service import FaissService
from src.environments.faiss.types import (
    FaissConfig,
    FaissSearchRequest,
    FaissAddRequest,
    FaissDeleteRequest
)
from src.logger import logger
from src.utils import assemble_project_path
from src.environments.protocol.server import ecp
from src.environments.protocol.environment import BaseEnvironment

@ecp.environment(
    name="faiss",
    type="Vector Store",
    description="FAISS vector store environment for similarity search and document management",
    has_vision=False,
    additional_rules={
        "state": "The state of the FAISS vector store environment.",
    }
)
class FaissEnvironment(BaseEnvironment):
    """FAISS Vector Store Environment that provides vector operations as an environment interface."""
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        embedding_function: Optional[Any] = None,
        config: Optional[FaissConfig] = None,
    ):
        """
        Initialize the FAISS environment.
        
        Args:
            base_dir: Base directory for FAISS storage
            embedding_function: Embedding function to use
            config: Configuration for the FAISS service
        """
        self.base_dir = Path(assemble_project_path(str(base_dir)))
        self.embedding_function = embedding_function
        self.config = config or FaissConfig(base_dir=str(self.base_dir))
        
        # Initialize FAISS service
        self.faiss_service = FaissService(
            base_dir=self.base_dir,
            embedding_function=self.embedding_function,
            config=self.config
        )
        
    async def initialize(self) -> None:
        """Initialize the FAISS environment."""
        logger.info(f"| 🔍 FAISS Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the FAISS environment."""
        await self.faiss_service.cleanup()
        logger.info("| 🧹 FAISS Environment cleanup completed")
    
    @ecp.action(
        name="add_documents",
        type="Faiss Environment",
        description="Add documents to the FAISS vector store",
    )
    async def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Add documents to the FAISS vector store.
        
        Args:
            texts (List[str]): List of texts to add to the vector store
            metadatas (Optional[List[Dict[str, Any]]]): Optional metadata for each text
        
        Returns:
            str: Count of added documents
        """
        request = FaissAddRequest(
            texts=texts,
            metadatas=metadatas,
        )
        
        result = await self.faiss_service.add_documents(request)
        
        return f"Successfully added {result.count} documents"
    
    @ecp.action(
        name="search_similar",
        description="Search for similar documents in the FAISS vector store",
        type="Faiss Environment",
        metadata={}
    )
    async def search_similar(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        score_threshold: Optional[float] = None
    ) -> str:
        """Search for similar documents in the FAISS vector store.
        
        Args:
            query (str): Query text to search for
            k (int): Number of documents to return (1-1000)
            filter (Optional[Dict[str, Any]]): Filter by metadata
            fetch_k (int): Number of documents to fetch before filtering (1-10000)
            score_threshold (Optional[float]): Minimum similarity score (0.0-1.0)
            
        Returns:
            str: Search results with documents and scores
        """
        request = FaissSearchRequest(
            query=query,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            score_threshold=score_threshold
        )
        
        result = await self.faiss_service.search_similar(request)
        
        # Format results as string
        if result.documents:
            documents_info = []
            for i, (doc, score) in enumerate(zip(result.documents, result.scores)):
                documents_info.append(f"Document {i+1} (Score: {score:.4f}):\n{doc.page_content[:200]}...")
            
            return f"Found {result.total_found} similar documents for query '{query}':\n\n" + "\n\n".join(documents_info)
        else:
            return f"No similar documents found for query '{query}'"
    
    @ecp.action(
        name="delete_documents",
        description="Delete documents from the FAISS vector store",
        type="Faiss Environment",
    )
    async def delete_documents(self, ids: List[str]) -> str:
        """Delete documents from the FAISS vector store.
        
        Args:
            ids (List[str]): IDs of documents to delete
            
        Returns:
            str: Deletion result message
        """
        request = FaissDeleteRequest(ids=ids)
        result = await self.faiss_service.delete_documents(request)
        
        if result.success:
            return f"Successfully deleted {result.deleted_count} documents"
        else:
            return f"Failed to delete documents. Deleted count: {result.deleted_count}"
    
    @ecp.action(
        name="get_index_info",
        description="Get information about the FAISS index",
        type="Faiss Environment"
    )
    async def get_index_info(self) -> str:
        """Get information about the FAISS index.
        
        Returns:
            str: Index information including document count, dimensions, and configuration
        """
        result = await self.faiss_service.get_index_info()
        
        return f"FAISS Index Information:\n" \
               f"Total Documents: {result.total_documents}\n" \
               f"Embedding Dimension: {result.embedding_dimension}\n" \
               f"Index Type: {result.index_type}\n" \
               f"Distance Strategy: {result.distance_strategy}"
    
    @ecp.action(
        name="save_index",
        description="Save the FAISS index to disk",
        type="Faiss Environment"
    )
    async def save_index(self) -> str:
        """Save the FAISS index to disk.
        
        Returns:
            str: Save operation result message
        """
        await self.faiss_service.save_index()
        return f"FAISS index saved successfully to: {self.base_dir}"
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the FAISS environment.
        
        Returns:
            Dict[str, Any]: Environment state including index information and configuration
        """
        try:
            index_info = await self.faiss_service.get_index_info()
            return {
                "base_dir": str(self.base_dir),
                "index_name": self.config.index_name,
                "total_documents": index_info.total_documents,
                "embedding_dimension": index_info.embedding_dimension,
                "index_type": index_info.index_type,
                "distance_strategy": index_info.distance_strategy,
                "auto_save": self.config.auto_save,
                "save_interval": self.config.save_interval
            }
        except Exception as e:
            logger.error(f"| ❌ Failed to get FAISS state: {e}")
            return {
                "base_dir": str(self.base_dir),
                "error": str(e)
            }
