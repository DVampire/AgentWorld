"""System prompt management for agents."""

from typing import Dict, Any
from langchain_core.messages import SystemMessage

from src.logger import logger
from src.agents.prompts.templates import PROMPT_TEMPLATES
from src.optimizers.protocol.variable import Variable

class SystemPrompt:
    """System prompt manager for tool-calling agents (static constitution).

    Parameters
    - prompt_name: selects a template entry from PROMPT_TEMPLATES if present (default: "tool_calling_system_prompt")
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
        try:
            prompt = PROMPT_TEMPLATES[self.prompt_name]
            self.prompt = Variable.from_dict(prompt)
            
            modules = self.prompt.get_modules()
            
            prompt_str = self.prompt.render(modules)
            
            self.message = SystemMessage(content=prompt_str, cache=True)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load system prompt: {e}")

    def get_message(self, 
                    modules: Dict[str, Any] = None,
                    reload: bool = False, 
                    **kwargs) -> SystemMessage:
        """Get the system prompt for the agent."""
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
