"""Tool Context Manager for managing tool lifecycle and resources."""

import atexit
from typing import Any, Dict, Callable, List
import importlib
import asyncio

from src.logger import logger
from src.infrastructures.models import model_manager
from src.environments.faiss.service import FaissService
from src.environments.faiss.types import FaissAddRequest, FaissSearchRequest
from src.utils import assemble_project_path
from src.tools.protocol.types import ToolInfo

class ToolContextManager:
    """Global context manager for all tools."""
    
    def __init__(self):
        """Initialize the tool context manager."""
        self._tool_info: Dict[str, ToolInfo] = {}
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
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.ainvoke(name, input, **kwargs))
            finally:
                loop.close()
        except Exception as e:
            return f"Error in synchronous execution: {str(e)}"
    
    async def ainvoke(self, name: str, input: Any, **kwargs) -> Any:
        """Invoke a tool.
        
        Args:
            name: Name of the tool
            input: Input for the tool
            **kwargs: Keyword arguments for the tool
        """
        if name in self._tool_info:
            instance = self._tool_info[name].instance
            return await instance.ainvoke(input, **kwargs)
        else:
            raise ValueError(f"Tool {name} not found")
    
    async def build(self, tool_info: ToolInfo, tool_factory: Callable) -> ToolInfo:
        """Create a tool instance and store it.
        
        Args:
            tool_info: Tool information
            tool_factory: Function to create the tool instance
            
        Returns:
            ToolInfo: Tool information
        """
        if tool_info.name in self._tool_info:
            return self._tool_info[tool_info.name]
        
        # Create new tool instance
        try:
            tool_instance = tool_factory()
            
            tool_info.instance = tool_instance
            
            # Store tool metadata
            self._tool_info[tool_info.name] = tool_info
            
            logger.debug(f"| 🔧 Tool {tool_info.name} created and stored")
            
            # Add tool to embedding index
            await self._store(tool_info)
            
            return tool_info
        except Exception as e:
            logger.error(f"| ❌ Failed to create tool {tool_info.name}: {e}")
            raise
    
    async def _store(self, tool_info: ToolInfo):
        """Add tool information to the embedding index.
        
        Args:
            tool_info: Tool information
        """
        try:
            # Create comprehensive text representation
            tool_text = f"Tool: {tool_info.name}\nType: {tool_info.type}\nDescription: {tool_info.description}"
            
            # Add to FAISS index
            request = FaissAddRequest(
                texts=[tool_text],
                metadatas=[{
                    "name": tool_info.name,
                    "type": tool_info.type,
                    "description": tool_info.description
                }]
            )
            
            await self._faiss_service.add_documents(request)
            logger.info(f"| 📝 Tool {tool_info.name} added to FAISS index")
            
        except Exception as e:
            logger.warning(f"| ⚠️ Failed to add tool {tool_info.name} to FAISS index: {e}")
    
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
            
                logger.info(f"| 🔍 Found {len(candidates)} tool candidates for query: {query}")
            return candidates
            
        except Exception as e:
            logger.error(f"| ❌ Failed to get tool candidates: {e}")
            return []
    
    def cleanup(self):
        """Cleanup all active tools."""
        try:
            # Get list of active tool names to avoid dict modification during iteration
            self._tool_info.clear()
                
            # Clean up Faiss service
            self._faiss_service.cleanup()
            logger.info("| 🧹 Tool context manager cleaned up")
            
        except Exception as e:
            logger.error(f"| ❌ Error during tool context manager cleanup: {e}")
