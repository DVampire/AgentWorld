"""Base model class for all language models."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from langchain_core.language_models import BaseLanguageModel
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import asyncio

from src.utils import Singleton


class BaseModel(ABC):
    """Base class for all language models."""
    
    def __init__(
        self,
        name: str,
        model_type: str,
        config: Dict[str, Any],
        **kwargs
    ):
        self.name = name
        self.model_type = model_type
        self.config = config
        self.llm = None
        self._setup_model()
    
    @abstractmethod
    def _setup_model(self):
        """Setup the language model."""
        pass
    
    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs) -> AIMessage:
        """Invoke the model asynchronously."""
        pass
    
    @abstractmethod
    async def agenerate(self, messages: List[BaseMessage], **kwargs) -> List[AIMessage]:
        """Generate responses for multiple messages asynchronously."""
        pass
    
    def invoke(self, prompt: str, **kwargs) -> AIMessage:
        """Invoke the model synchronously (wrapper for async)."""
        return asyncio.run(self.ainvoke(prompt, **kwargs))
    
    def generate(self, messages: List[BaseMessage], **kwargs) -> List[AIMessage]:
        """Generate responses for multiple messages synchronously."""
        return asyncio.run(self.agenerate(messages, **kwargs))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "name": self.name,
            "type": self.model_type,
            "config": self.config,
            "llm": self.llm
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update model configuration."""
        self.config.update(new_config)
        self._setup_model()  # Recreate model with new config
    
    def get_config(self) -> Dict[str, Any]:
        """Get current model configuration."""
        return self.config.copy()
    
    def is_async(self) -> bool:
        """Check if the model supports async operations."""
        return True  # All our models are async
    
    def get_max_tokens(self) -> Optional[int]:
        """Get maximum tokens for the model."""
        return self.config.get("max_tokens")
    
    def get_temperature(self) -> float:
        """Get temperature setting for the model."""
        return self.config.get("temperature", 0.7)
    
    def get_model_name(self) -> str:
        """Get the underlying model name."""
        return self.config.get("model_name", self.name)
