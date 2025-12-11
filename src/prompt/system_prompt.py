"""System prompt management for agents."""

from typing import Dict, Any
from langchain_core.messages import SystemMessage

from src.logger import logger
from src.prompt.manager import prompt_manager
from src.optimizer.protocol.variable import Variable

class SystemPrompt:
    """System prompt manager for tool-calling agents (static constitution).

    Parameters
    - prompt_name: selects a template entry from prompt_manager if present (default: "tool_calling_system_prompt")
    - max_tools: number used to format {{ max_tools }} in the template
    """

    def __init__(
        self,
        prompt_name: str = "tool_calling_system_prompt",
        **kwargs
    ):
        self.prompt_name = prompt_name
        
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the system prompt."""
        # Note: prompt_manager.get() is async, but we can't use await in __init__
        # So we'll defer the actual loading until get_message() is called
        self.prompt = None
        self.message = None
    
    async def _load_prompt(self) -> None:
        """Load prompt template asynchronously."""
        if self.prompt is not None:
            return
        
        try:
            # Get prompt template from prompt_manager
            prompt_dict = await prompt_manager.get(self.prompt_name)
            if not prompt_dict:
                # Fallback to template module for backward compatibility
                from src.prompt.template import PROMPT_TEMPLATES
                prompt_dict = PROMPT_TEMPLATES.get(self.prompt_name)
                if not prompt_dict:
                    raise ValueError(f"Prompt {self.prompt_name} not found")
            
            self.prompt = Variable.from_dict(prompt_dict)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load system prompt: {e}")

    async def get_message(self, 
                    modules: Dict[str, Any] = None,
                    reload: bool = False, 
                    **kwargs) -> SystemMessage:
        """Get the system prompt for the agent."""
        # Load prompt if not already loaded or if reloading
        if self.prompt is None or reload:
            await self._load_prompt()
        
        if not reload and self.message is not None:
            return self.message
        try:
            
            modules = modules if modules is not None else {}
            
            prompt_str = self.prompt.render(modules)
            
            self.message = SystemMessage(content=prompt_str, cache=True)
            
        except Exception as e:
            logger.warning(f"Failed to render system prompt: {e}")
            raise RuntimeError(f"Failed to render system prompt: {e}")
        return self.message
