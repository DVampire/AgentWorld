"""FAISS Vector Store Environment for AgentWorld."""

from typing import Any, Dict, List, Union, Optional, Any, Dict, Type
from pydantic import BaseModel, Field, ConfigDict

from src.environment.faiss.service import FaissService
from src.environment.faiss.types import (
    FaissConfig,
    FaissSearchRequest,
    FaissAddRequest,
    FaissDeleteRequest
)
from src.utils import dedent
from src.logger import logger
from src.utils import assemble_project_path
from src.environment.server import environment_manager
from src.environment.types import Environment
from src.registry import ENVIRONMENT

@ENVIRONMENT.register_module(force=True)
class FaissEnvironment(Environment):
    """FAISS Vector Store Environment that provides vector operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="faiss", description="The name of the FAISS environment.")
    description: str = Field(default="FAISS vector store environment for similarity search and document management", description="The description of the FAISS environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the FAISS vector store environment.",
        }
    }, description="The metadata of the FAISS environment.")
    require_grad: bool = Field(default=False, description="Whether the environment requires gradients")
    
    def __init__(
        self,
        base_dir: str,
        model_name: Optional[str] = None,
        config: Optional[FaissConfig] = None,
        require_grad: bool = False,
        **kwargs
    ):
        """
        Initialize the FAISS environment.
        
        Args:
            base_dir (str): Base directory for FAISS storage
            model_name (Optional[str]): Embedding model name to use
            config (Optional[FaissConfig]): Configuration for the FAISS service
        """
        super().__init__(**kwargs)
        
        self.base_dir = assemble_project_path(base_dir)
        self.model_name = model_name
        self.config = config or FaissConfig(base_dir=self.base_dir)
        
        # Initialize FAISS service
        self.faiss_service = FaissService(
            base_dir=self.base_dir,
            model_name=self.model_name or self.config.model_name,
            config=self.config
        )
        
    async def initialize(self) -> None:
        """Initialize the FAISS environment."""
        logger.info(f"| 🔍 FAISS Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the FAISS environment."""
        await self.faiss_service.cleanup()
        logger.info("| 🧹 FAISS Environment cleanup completed")
    
    @environment_manager.action(
        name="add_documents",
        description="Add documents to the FAISS vector store",
    )
    async def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Add documents to the FAISS vector store.
        
        Args:
            texts (List[str]): List of texts to add to the vector store
            metadatas (Optional[List[Dict[str, Any]]]): Optional metadata for each text
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FaissAddRequest(
                texts=texts,
                metadatas=metadatas,
            )
            
            result = await self.faiss_service.add_documents(request)
            
            extra = result.extra.copy() if result.extra else {}
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to add documents: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    @environment_manager.action(
        name="search_similar",
        description="Search for similar documents in the FAISS vector store",
        metadata={}
    )
    async def search_similar(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: int = 20,
        score_threshold: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Search for similar documents in the FAISS vector store.
        
        Args:
            query (str): Query text to search for
            k (int): Number of documents to return (1-1000)
            filter (Optional[Dict[str, Any]]): Filter by metadata
            fetch_k (int): Number of documents to fetch before filtering (1-10000)
            score_threshold (Optional[float]): Minimum similarity score (0.0-1.0)
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FaissSearchRequest(
                query=query,
                k=k,
                filter=filter,
                fetch_k=fetch_k,
                score_threshold=score_threshold
            )
            
            result = await self.faiss_service.search_similar(request)
            
            extra = result.extra.copy() if result.extra else {}
            
            # Format message from results
            if result.success and "documents" in extra and extra["documents"]:
                documents_info = []
                documents = extra["documents"]
                scores = extra.get("scores", [])
                for i, (doc, score) in enumerate(zip(documents, scores)):
                    content = doc.get("page_content", "")[:200] if isinstance(doc, dict) else str(doc)[:200]
                    documents_info.append(f"Document {i+1} (Score: {score:.4f}):\n{content}...")
                message = f"Found {extra.get('total_found', 0)} similar documents for query '{query}':\n\n" + "\n\n".join(documents_info)
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
                "message": f"Failed to search documents: {str(e)}",
                "extra": {"error": str(e), "query": query}
            }
    
    @environment_manager.action(
        name="delete_documents",
        description="Delete documents from the FAISS vector store",
    )
    async def delete_documents(self, ids: List[str], **kwargs) -> Dict[str, Any]:
        """Delete documents from the FAISS vector store.
        
        Args:
            ids (List[str]): IDs of documents to delete
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FaissDeleteRequest(ids=ids)
            result = await self.faiss_service.delete_documents(request)
            
            extra = result.extra.copy() if result.extra else {}
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete documents: {str(e)}",
                "extra": {"error": str(e), "ids": ids}
            }
    
    @environment_manager.action(
        name="get_index_info",
        description="Get information about the FAISS index"
    )
    async def get_index_info(self, **kwargs) -> Dict[str, Any]:
        """Get information about the FAISS index.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            result = await self.faiss_service.get_index_info()
            
            extra = result.extra.copy() if result.extra else {}
            
            if result.success:
                index_info = extra.get("index_info", {})
                message = f"FAISS Index Information:\n" \
                         f"Total Documents: {extra.get('total_documents', 0)}\n" \
                         f"Embedding Dimension: {extra.get('embedding_dimension', 0)}\n" \
                         f"Index Type: {extra.get('index_type', 'Unknown')}\n" \
                         f"Distance Strategy: {extra.get('distance_strategy', 'Unknown')}"
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
                "message": f"Failed to get index info: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    @environment_manager.action(
        name="save_index",
        description="Save the FAISS index to disk"
    )
    async def save_index(self, **kwargs) -> Dict[str, Any]:
        """Save the FAISS index to disk.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            await self.faiss_service.save_index()
            return {
                "success": True,
                "message": f"FAISS index saved successfully to: {self.base_dir}",
                "extra": {"base_dir": str(self.base_dir)}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to save index: {str(e)}",
                "extra": {"error": str(e), "base_dir": str(self.base_dir)}
            }
    
    async def get_state(self, **kwargs) -> Dict[str, Any]:
        """Get the current state of the FAISS environment.
        
        Returns:
            Dict[str, Any]: Environment state including index information and configuration
        """
        try:
            index_result = await self.faiss_service.get_index_info()
            extra = index_result.extra if index_result.extra else {}
            
            state = dedent(f"""
                <info>
                Base Directory: {str(self.base_dir)}
                Index Name: {self.config.index_name}
                Total Documents: {extra.get("total_documents", 0)}
                Embedding Dimension: {extra.get("embedding_dimension", 0)}
                Index Type: {extra.get("index_type", "Unknown")}
                Distance Strategy: {extra.get("distance_strategy", "Unknown")}
                Auto Save: {self.config.auto_save}
                Save Interval: {self.config.save_interval}
                </info>
            """
            )
            return {
                "state": state,
                "extra": extra
            }
        except Exception as e:
            logger.error(f"| ❌ Failed to get FAISS state: {e}")
            return {
                "state": f"Failed to get FAISS state: {str(e)}",
                "extra": {"error": str(e)}
            }
