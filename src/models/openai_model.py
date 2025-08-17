"""OpenAI async model implementation."""

from typing import Dict, List, Any, Optional
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
import os

from .base_model import BaseModel


class OpenAIAsyncModel(BaseModel):
    """OpenAI async model wrapper."""
    
    def __init__(
        self,
        name: str,
        model_name: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        config = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": api_key or os.getenv("OPENAI_API_KEY"),
            **kwargs
        }
        
        super().__init__(name, "openai", config)
    
    def _setup_model(self):
        """Setup the OpenAI model."""
        try:
            self.llm = ChatOpenAI(
                model=self.config["model_name"],
                temperature=self.config["temperature"],
                max_tokens=self.config.get("max_tokens"),
                openai_api_key=self.config["api_key"],
                streaming=self.config.get("streaming", False),
                verbose=self.config.get("verbose", False)
            )
        except Exception as e:
            print(f"Error setting up OpenAI model: {e}")
            raise
    
    async def ainvoke(self, prompt: str, **kwargs) -> AIMessage:
        """Invoke the OpenAI model asynchronously."""
        if not self.llm:
            raise RuntimeError("OpenAI model not initialized")
        
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
            print(f"Error invoking OpenAI model: {e}")
            raise
    
    async def agenerate(self, messages: List[BaseMessage], **kwargs) -> List[AIMessage]:
        """Generate responses for multiple messages asynchronously."""
        if not self.llm:
            raise RuntimeError("OpenAI model not initialized")
        
        try:
            # Generate responses for all messages
            responses = await self.llm.agenerate([messages])
            
            # Convert to AIMessage list
            ai_messages = []
            for response in responses.generations[0]:
                ai_messages.append(AIMessage(content=response.text))
            
            return ai_messages
            
        except Exception as e:
            print(f"Error generating with OpenAI model: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the OpenAI model."""
        info = super().get_model_info()
        info.update({
            "provider": "OpenAI",
            "model_name": self.config["model_name"],
            "temperature": self.config["temperature"],
            "max_tokens": self.config.get("max_tokens"),
            "streaming": self.config.get("streaming", False)
        })
        return info
    
    @classmethod
    def create_gpt4(cls, name: str = "gpt-4", **kwargs) -> "OpenAIAsyncModel":
        """Create a GPT-4 model instance."""
        return cls(name, model_name="gpt-4", **kwargs)
    
    @classmethod
    def create_gpt35_turbo(cls, name: str = "gpt-3.5-turbo", **kwargs) -> "OpenAIAsyncModel":
        """Create a GPT-3.5 Turbo model instance."""
        return cls(name, model_name="gpt-3.5-turbo", **kwargs)
    
    @classmethod
    def create_gpt4_turbo(cls, name: str = "gpt-4-turbo-preview", **kwargs) -> "OpenAIAsyncModel":
        """Create a GPT-4 Turbo model instance."""
        return cls(name, model_name="gpt-4-turbo-preview", **kwargs)
