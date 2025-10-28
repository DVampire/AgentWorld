"""Agent message prompt management for dynamic task-related prompts."""

import os
from typing import Dict, Any
from langchain_core.messages import HumanMessage
from bs4 import BeautifulSoup
from PIL import Image

from src.logger import logger
from src.agents.prompts.templates import PROMPT_TEMPLATES
from src.utils import encode_image_base64, make_image_url, assemble_project_path
from src.optimizers.protocol.variable import Variable

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
        try:
            prompt = PROMPT_TEMPLATES[self.prompt_name]
            self.prompt = Variable.from_dict(prompt)
        except Exception as e:
            raise RuntimeError(f"Failed to load agent message prompt: {e}")
        self.message = None
    
    def get_message(self,
                    modules: Dict[str, Any] = None,
                    reload: bool = True, 
                    **kwargs) -> HumanMessage:
        """Get complete task state as a single message using template."""
        try:
            
            modules = modules if modules is not None else {}
            
            prompt_str = self.prompt.render(modules)
            
            contents = [
                {"type": "text", "text": prompt_str},
            ]
            
            soup = BeautifulSoup(prompt_str, 'html.parser')
            images = soup.find_all('img')
            
            for image in images:
                image_path = image.get('src')
                image_path = assemble_project_path(image_path)
                image_instruction = image.get('alt')
                
                if os.path.exists(image_path):
                    try:
                        contents.append({
                            "type": "text",
                            "text": image_instruction
                        })
                        
                        image = Image.open(image_path)
                        image_url = make_image_url(encode_image_base64(image))
                        contents.append({
                            "type": "image_url", 
                            "image_url": {"url": image_url}
                        })
                    except Exception as e:
                        logger.warning(f"Failed to process image {image_path}: {e}")
            
            return HumanMessage(content=contents, cache=True)
            
        except Exception as e:
            logger.warning(f"Failed to render agent message template: {e}")
            raise RuntimeError(f"Failed to render agent message template: {e}")