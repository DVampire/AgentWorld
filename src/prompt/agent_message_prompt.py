"""Agent message prompt management for dynamic task-related prompts."""

import os
from typing import Dict, Any
from bs4 import BeautifulSoup
from PIL import Image

from src.logger import logger
from src.prompt.manager import prompt_manager
from src.utils import make_file_url, assemble_project_path
from src.message import HumanMessage, ImageURL, ContentPartImage, ContentPartText
from src.optimizer.protocol.variable import Variable

class AgentMessagePrompt:
    """Agent message prompt manager for dynamic task-related prompts (tool-calling agents)."""
    
    def __init__(
        self,
        prompt_name: str = "tool_calling_agent_message_prompt",
        **kwargs
    ):
        self.prompt_name = prompt_name
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the agent message prompt."""
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
                available = await prompt_manager.list()
                raise ValueError(f"Prompt {self.prompt_name} not found. Available prompts: {available}")
            
            self.prompt = Variable.from_dict(prompt_dict)
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