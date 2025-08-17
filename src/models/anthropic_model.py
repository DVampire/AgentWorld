"""Anthropic async model implementation."""

from typing import Dict, List, Any, Optional
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
import os

from .base_model import BaseModel


class AnthropicAsyncModel(BaseModel):
    """Anthropic async model wrapper."""
    
    def __init__(
        self,
        name: str,
        model_name: str = "claude-3-sonnet-20240229",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        config = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": api_key or os.getenv("ANTHROPIC_API_KEY"),
            **kwargs
        }
        
        super().__init__(name, "anthropic", config)
    
    def _setup_model(self):
        """Setup the Anthropic model."""
        try:
            self.llm = ChatAnthropic(
                model=self.config["model_name"],
                temperature=self.config["temperature"],
                max_tokens=self.config.get("max_tokens"),
                anthropic_api_key=self.config["api_key"],
                streaming=self.config.get("streaming", False),
                verbose=self.config.get("verbose", False)
            )
        except Exception as e:
            print(f"Error setting up Anthropic model: {e}")
            raise
    
    async def ainvoke(self, prompt: str, **kwargs) -> AIMessage:
        """Invoke the Anthropic model asynchronously."""
        if not self.llm:
            raise RuntimeError("Anthropic model not initialized")
        
        try:
            # Create a human message from the prompt
            message = HumanMessage(content=prompt)
            
            # Invoke the model
            response = await self.llm.ainvoke([message])
            
            # Return as AIMessage
            if hasattr(response, 'content'):
                return AIMessage(content=response.content)
            else:
                return AIMessage(content=str(response))
                
        except Exception as e:
            print(f"Error invoking Anthropic model: {e}")
            raise
    
    async def agenerate(self, messages: List[BaseMessage], **kwargs) -> List[AIMessage]:
        """Generate responses for multiple messages asynchronously."""
        if not self.llm:
            raise RuntimeError("Anthropic model not initialized")
        
        try:
            # Generate responses for all messages
            responses = await self.llm.agenerate([messages])
            
            # Convert to AIMessage list
            ai_messages = []
            for response in responses.generations[0]:
                ai_messages.append(AIMessage(content=response.text))
            
            return ai_messages
            
        except Exception as e:
            print(f"Error generating with Anthropic model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the Anthropic model."""
        info = super().get_model_info()
        info.update({
            "provider": "Anthropic",
            "model_name": self.config["model_name"],
            "temperature": self.config["temperature"],
            "max_tokens": self.config.get("max_tokens"),
            "streaming": self.config.get("streaming", False)
        })
        return info
    
    @classmethod
    def create_claude3_sonnet(cls, name: str = "claude-3-sonnet", **kwargs) -> "AnthropicAsyncModel":
        """Create a Claude 3 Sonnet model instance."""
        return cls(name, model_name="claude-3-sonnet-20240229", **kwargs)
    
    @classmethod
    def create_claude3_opus(cls, name: str = "claude-3-opus", **kwargs) -> "AnthropicAsyncModel":
        """Create a Claude 3 Opus model instance."""
        return cls(name, model_name="claude-3-opus-20240229", **kwargs)
    
    @classmethod
    def create_claude3_haiku(cls, name: str = "claude-3-haiku", **kwargs) -> "AnthropicAsyncModel":
        """Create a Claude 3 Haiku model instance."""
        return cls(name, model_name="claude-3-haiku-20240307", **kwargs)
    
    @classmethod
    def create_claude2(cls, name: str = "claude-2", **kwargs) -> "AnthropicAsyncModel":
        """Create a Claude 2 model instance."""
        return cls(name, model_name="claude-2.1", **kwargs)
