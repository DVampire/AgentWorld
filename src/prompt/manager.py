"""Prompt Manager

Manager implementation for the Prompt Context Protocol.
"""

import asyncio
import os
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.utils import assemble_project_path
from src.logger import logger
from src.prompt.types import PromptConfig, Prompt
from src.prompt.context import PromptContextManager

class PromptManager(BaseModel):
    """Prompt Manager for managing prompt registration and lifecycle"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the prompts")
    save_path: str = Field(default=None, description="The path to save the prompts")
    
    def __init__(self, **kwargs):
        """Initialize the Prompt Manager."""
        super().__init__(**kwargs)
        self._registered_prompts: Dict[str, PromptConfig] = {}  # prompt_name -> PromptConfig
    
    async def initialize(self):
        """Initialize prompts by discovering and registering them."""
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "prompts"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "prompts.json")
        logger.info(f"| 📁 Prompt Manager base directory: {self.base_dir} and save path: {self.save_path}")
        
        # Initialize prompt context manager (this will trigger discovery if auto_discover is True)
        self.prompt_context_manager = PromptContextManager(base_dir=self.base_dir, save_path=self.save_path)
        await self.prompt_context_manager.initialize()
        
        # Sync registered_prompts from context manager after discovery
        prompt_names = await self.prompt_context_manager.list()
        for prompt_name in prompt_names:
            prompt_config = await self.prompt_context_manager.get_info(prompt_name)
            if prompt_config and prompt_name not in self._registered_prompts:
                self._registered_prompts[prompt_name] = prompt_config
        
        logger.info("| ✅ Prompts initialization completed")
    
    async def register(self, prompt: Union[Prompt, Dict[str, Any]], *, override: bool = False, **kwargs: Any) -> PromptConfig:
        """Register a prompt or prompt template dictionary asynchronously.
        
        Args:
            prompt: Prompt instance or template dictionary to register
            override: Whether to override existing registration
            **kwargs: Configuration for prompt initialization
            
        Returns:
            PromptConfig: Prompt configuration
        """
        prompt_config = await self.prompt_context_manager.register(prompt, override=override, **kwargs)
        self._registered_prompts[prompt_config.name] = prompt_config
        return prompt_config
    
    async def update(self, prompt_name: str, prompt: Union[Prompt, Dict[str, Any]], 
                    new_version: Optional[str] = None, description: Optional[str] = None,
                    **kwargs: Any) -> PromptConfig:
        """Update an existing prompt with new configuration and create a new version
        
        Args:
            prompt_name: Name of the prompt to update
            prompt: New prompt instance or template dictionary with updated content
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update
            **kwargs: Configuration for prompt initialization
            
        Returns:
            PromptConfig: Updated prompt configuration
        """
        prompt_config = await self.prompt_context_manager.update(prompt_name, prompt, new_version, description, **kwargs)
        self._registered_prompts[prompt_config.name] = prompt_config
        return prompt_config
    
    async def get_info(self, prompt_name: str) -> Optional[PromptConfig]:
        """Get prompt configuration by name
        
        Args:
            prompt_name: Prompt name
            
        Returns:
            PromptConfig: Prompt configuration or None if not found
        """
        return await self.prompt_context_manager.get_info(prompt_name)
    
    async def list(self) -> List[str]:
        """List all registered prompts
        
        Returns:
            List[str]: List of prompt names
        """
        return await self.prompt_context_manager.list()
    
    async def get(self, prompt_name: str) -> Optional[Dict[str, Any]]:
        """Get prompt template dictionary by name
        
        Args:
            prompt_name: Prompt name
            
        Returns:
            Dict[str, Any]: Prompt template dictionary or None if not found
        """
        return await self.prompt_context_manager.get(prompt_name)
    
    async def save_to_json(self, file_path: Optional[str] = None) -> str:
        """Save all prompt configurations to JSON
        
        Args:
            file_path: File path to save to
            
        Returns:
            Path to saved file
        """
        file_path = file_path if file_path is not None else self.save_path
        return await self.prompt_context_manager.save_to_json(file_path)
    
    async def load_from_json(self, file_path: Optional[str] = None) -> bool:
        """Load prompt configurations from JSON
        
        Args:
            file_path: File path to load from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        file_path = file_path if file_path is not None else self.save_path
        success = await self.prompt_context_manager.load_from_json(file_path)
        if success:
            # Sync registered_prompts
            prompt_names = await self.prompt_context_manager.list()
            for prompt_name in prompt_names:
                prompt_config = await self.prompt_context_manager.get_info(prompt_name)
                if prompt_config:
                    self._registered_prompts[prompt_name] = prompt_config
        return success
    
    async def get_system_message(self, 
                          prompt_name: Optional[str] = None,
                          modules: Dict[str, Any] = None, 
                          reload: bool = False, 
                          **kwargs):
        """Get a system message using SystemPrompt.
        
        Args:
            prompt_name: Name of the prompt (e.g., "tool_calling_system_prompt"). 
                        If None, will try to infer from kwargs or use default.
            modules: Modules to render in the template
            reload: Whether to reload the prompt
            **kwargs: Additional arguments (may include prompt_name for backward compatibility)
        """
        from src.prompt.system_prompt import SystemPrompt
        
        # Ensure prompt_manager is initialized
        if not hasattr(self, 'prompt_context_manager'):
            await self.initialize()
        
        # Support backward compatibility: if prompt_name not provided, try to get from kwargs or use default
        if prompt_name is None:
            prompt_name = kwargs.pop('prompt_name', None)
            if prompt_name is None:
                # Try to infer from prompt_name in modules or use default
                if modules and 'prompt_name' in modules:
                    base_name = modules['prompt_name']
                    prompt_name = f"{base_name}_system_prompt"
                else:
                    prompt_name = "tool_calling_system_prompt"
        
        # SystemPrompt will handle getting the template (from prompt_manager or fallback to PROMPT_TEMPLATES)
        system_prompt = SystemPrompt(prompt_name=prompt_name, **kwargs)
        return await system_prompt.get_message(modules, reload, **kwargs)
    
    async def get_agent_message(self, 
                         prompt_name: Optional[str] = None,
                         modules: Dict[str, Any] = None, 
                         reload: bool = True, 
                         **kwargs):
        """Get an agent message using AgentMessagePrompt.
        
        Args:
            prompt_name: Name of the prompt (e.g., "tool_calling_agent_message_prompt").
                        If None, will try to infer from kwargs or use default.
            modules: Modules to render in the template
            reload: Whether to reload the prompt
            **kwargs: Additional arguments (may include prompt_name for backward compatibility)
        """
        from src.prompt.agent_message_prompt import AgentMessagePrompt
        
        # Ensure prompt_manager is initialized
        if not hasattr(self, 'prompt_context_manager'):
            await self.initialize()
        
        # Support backward compatibility: if prompt_name not provided, try to get from kwargs or use default
        if prompt_name is None:
            prompt_name = kwargs.pop('prompt_name', None)
            if prompt_name is None:
                # Try to infer from prompt_name in modules or use default
                if modules and 'prompt_name' in modules:
                    base_name = modules['prompt_name']
                    prompt_name = f"{base_name}_agent_message_prompt"
                else:
                    prompt_name = "tool_calling_agent_message_prompt"
        
        # AgentMessagePrompt will handle getting the template (from prompt_manager or fallback to PROMPT_TEMPLATES)
        agent_message_prompt = AgentMessagePrompt(prompt_name=prompt_name, **kwargs)
        return await agent_message_prompt.get_message(modules, reload, **kwargs)
    
    async def cleanup(self):
        """Cleanup all prompts using context manager."""
        if hasattr(self, 'prompt_context_manager'):
            await self.prompt_context_manager.cleanup()


# Global Prompt Manager instance
prompt_manager = PromptManager()
