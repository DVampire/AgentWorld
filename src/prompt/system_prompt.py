"""System prompt management for agents."""

from typing import Dict, Any, Optional

from src.logger import logger
from src.message import SystemMessage
from src.optimizer.protocol.variable import Variable
from src.prompt.types import PromptConfig

class SystemPrompt:
    """System prompt manager for tool-calling agents (static constitution).

    Parameters
    - prompt_config: PromptConfig or prompt dictionary to use for rendering
    """

    def __init__(
        self,
        prompt_config: Optional[PromptConfig] = None,
        prompt_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize SystemPrompt with either PromptConfig or prompt dictionary.
        
        Args:
            prompt_config: PromptConfig instance
            prompt_dict: Prompt dictionary (alternative to prompt_config)
            **kwargs: Additional arguments (ignored, kept for backward compatibility)
        """
        if prompt_config is not None:
            self.prompt_config = prompt_config
            # Convert PromptConfig to dict for Variable.from_dict
            self.prompt_dict = {
                "name": prompt_config.name,
                "type": prompt_config.type,
                "description": prompt_config.description,
                "template": prompt_config.template,
                "variables": prompt_config.variables,
                "metadata": prompt_config.metadata,
            }
        elif prompt_dict is not None:
            self.prompt_config = None
            self.prompt_dict = prompt_dict
        else:
            raise ValueError("Either prompt_config or prompt_dict must be provided")
        
        self._initialize()

    def _initialize(self) -> None:
        """Initialize the system prompt."""
        self.prompt = None
        self.message = None
    
    async def _load_prompt(self) -> None:
        """Load prompt template asynchronously."""
        if self.prompt is not None:
            return
        
        try:
            self.prompt = Variable.from_dict(self.prompt_dict)
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
            # Build modules from variable tree if not provided
            if modules is None or len(modules) == 0:
                modules = self.prompt.get_modules()
            else:
                # Merge provided modules with variable tree modules
                variable_modules = self.prompt.get_modules()
                modules = {**variable_modules, **modules}
            
            prompt_str = self.prompt.render(modules)
            
            self.message = SystemMessage(content=prompt_str, cache=True)
            
        except Exception as e:
            logger.warning(f"Failed to render system prompt: {e}")
            raise RuntimeError(f"Failed to render system prompt: {e}")
        return self.message
