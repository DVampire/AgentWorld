"""Agent message prompt management for dynamic task-related prompts."""

import os
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup
from PIL import Image

from src.logger import logger
from src.utils import make_file_url, assemble_project_path
from src.message import HumanMessage, ImageURL, ContentPartImage, ContentPartText
from src.optimizer.types import Variable
from src.prompt.types import PromptConfig

class AgentMessagePrompt:
    """Agent message prompt manager for dynamic task-related prompts (tool-calling agents)."""
    
    def __init__(
        self,
        prompt_config: Optional[PromptConfig] = None,
        prompt_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize AgentMessagePrompt with either PromptConfig or prompt dictionary.
        
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
        """Initialize the agent message prompt."""
        self.prompt = None
        self.message = None
    
    async def _load_prompt(self) -> None:
        """Load prompt template asynchronously."""
        if self.prompt is not None:
            return
        
        try:
            self.prompt = Variable.from_dict(self.prompt_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load agent message prompt: {e}")
    
    async def get_message(self,
                    modules: Dict[str, Any] = None,
                    reload: bool = True, 
                    **kwargs) -> HumanMessage:
        """Get complete task state as a single message using template."""
        # Load prompt if not already loaded or if reloading
        if self.prompt is None or reload:
            await self._load_prompt()
        
        try:
            modules = modules if modules is not None else {}
            
            prompt_str = self.prompt.render(modules)
            
            contents = [
                ContentPartText(text=prompt_str),
            ]
            
            soup = BeautifulSoup(prompt_str, 'html.parser')
            images = soup.find_all('img')
            
            for image in images:
                image_path = image.get('src')
                image_path = assemble_project_path(image_path)
                image_instruction = image.get('alt')
                
                if os.path.exists(image_path):
                    try:
                        contents.append(ContentPartText(text=image_instruction))
                        
                        image_url = make_file_url(image_path)
                        contents.append(ContentPartImage(image_url=ImageURL(url=image_url)))
                    except Exception as e:
                        logger.warning(f"Failed to process image {image_path}: {e}")
            
            return HumanMessage(content=contents, cache=True)
            
        except Exception as e:
            logger.warning(f"Failed to render agent message template: {e}")
            raise RuntimeError(f"Failed to render agent message template: {e}")
