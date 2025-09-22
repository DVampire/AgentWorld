"""Tool Context Manager for managing tool lifecycle and resources."""

import atexit
import uuid
from typing import Any, Dict, Optional, Callable, List
import importlib

from src.logger import logger
from src.infrastructures.models import model_manager
from src.environments.faiss.service import FaissService
from src.environments.faiss.types import FaissAddRequest, FaissSearchRequest
from src.utils import assemble_project_path

class ToolContextManager:
    """Global context manager for all tools."""
    
    def __init__(self):
        """Initialize the tool context manager."""
        self._tool_instances: Dict[str, Any] = {}
        self._tool_info: Dict[str, Dict[str, Any]] = {}  # Store tool metadata
        self._cleanup_registered = False
        
        # Register cleanup on exit
        if not self._cleanup_registered:
            atexit.register(self.cleanup)
            self._cleanup_registered = True
            
        # Initialize Faiss service for tool embedding
        self._faiss_service = FaissService(
            base_dir=assemble_project_path(str(importlib.resources.files("src.tools.protocol"))),
            embedding_function=model_manager.get("text-embedding-3-large")
        )
            
    def invoke(self, name: str, input: Any, **kwargs) -> Any:
        """Invoke a tool.
        
        Args:
            name: Name of the tool
            input: Input for the tool
            **kwargs: Keyword arguments for the tool
        """
        
        if name in self._tool_instances:
            instance = self._tool_instances[name]
            return instance.invoke(input, **kwargs)
        else:
            raise ValueError(f"Tool {name} not found")
    
    async def ainvoke(self, name: str, input: Any, **kwargs) -> Any:
        """Invoke a tool.
        
        Args:
            name: Name of the tool
            input: Input for the tool
            **kwargs: Keyword arguments for the tool
        """
        if name in self._tool_instances:
            instance = self._tool_instances[name]
            return await instance.ainvoke(input, **kwargs)
        else:
            raise ValueError(f"Tool {name} not found")
    
    async def build(self, tool_name: str, tool_factory: Callable, tool_type: str = "unknown", description: str = "") -> Any:
        """Create a tool instance and store it.
        
        Args:
            tool_name: Name of the tool
            tool_factory: Function to create the tool instance
            tool_type: Type/category of the tool
            description: Description of the tool
            
        Returns:
            Tool instance
        """
        if tool_name in self._tool_instances:
            return self._tool_instances[tool_name]
        
        # Create new tool instance
        try:
            tool_instance = tool_factory()
            self._tool_instances[tool_name] = tool_instance
            
            # Store tool metadata
            self._tool_info[tool_name] = {
                "name": tool_name,
                "type": tool_type,
                "description": description,
                "instance": tool_instance
            }
            
            logger.debug(f"| 🔧 Tool {tool_name} created and stored")
            
            # Add tool to embedding index
            await self._store(tool_name, tool_type, description)
            
            return tool_instance
        except Exception as e:
            logger.error(f"| ❌ Failed to create tool {tool_name}: {e}")
            raise
    
    async def _store(self, tool_name: str, tool_type: str, description: str):
        """Add tool information to the embedding index.
        
        Args:
            tool_name: Name of the tool
            tool_type: Type/category of the tool
            description: Description of the tool
        """
        try:
            # Create comprehensive text representation
            tool_text = f"Tool: {tool_name}\nType: {tool_type}\nDescription: {description}"
            
            # Add to FAISS index
            request = FaissAddRequest(
                texts=[tool_text],
                metadatas=[{
                    "name": tool_name,
                    "type": tool_type,
                    "description": description
                }]
            )
            
            await self._faiss_service.add_documents(request)
            logger.debug(f"| 📝 Tool {tool_name} added to embedding index")
            
        except Exception as e:
            logger.warning(f"| ⚠️ Failed to add tool {tool_name} to embedding: {e}")
    
    async def candidates(self, query: str, k: int = 5, score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get tool candidates based on semantic similarity to query.
        
        Args:
            query: Search query
            k: Number of candidates to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of tool candidates with metadata and scores
        """
        try:
            # Search for similar tools
            request = FaissSearchRequest(
                query=query,
                k=k,
                score_threshold=score_threshold
            )
            
            result = await self._faiss_service.search_similar(request)
            
            # Format results
            candidates = []
            for i, doc in enumerate(result.documents):
                if i < len(result.scores):
                    candidate = {
                        "name": doc.metadata.get("name", "unknown"),
                        "type": doc.metadata.get("type", "unknown"),
                        "description": doc.metadata.get("description", ""),
                        "score": result.scores[i],
                        "content": doc.page_content
                    }
                    candidates.append(candidate)
            
            logger.debug(f"| 🔍 Found {len(candidates)} tool candidates for query: {query}")
            return candidates
            
        except Exception as e:
            logger.error(f"| ❌ Failed to get tool candidates: {e}")
            return []
    
    def cleanup(self):
        """Cleanup all active tools."""
        logger.info("| 🧹 Cleaning up all tools...")
        
        # Get list of active tool names to avoid dict modification during iteration
        active_tools = list(self._tool_instances.keys())
        
        for tool_name in active_tools:
            self._tool_instances.pop(tool_name, None)
            self._tool_info.pop(tool_name, None)
            
        # Clean up Faiss service
        self._faiss_service.cleanup()
        
        logger.info("| ✅ All tools cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
