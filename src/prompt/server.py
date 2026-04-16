"""Prompt Manager

Manager implementation for the Prompt Context Protocol with version management.
"""

import os
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.utils import assemble_project_path
from src.logger import logger
from src.prompt.types import PromptConfig, Prompt
from src.prompt.context import PromptContextManager
from src.message.types import Message
from src.optimizer.types import Variable

class PromptManagerServer(BaseModel):
    """Prompt Manager for managing prompt registration and lifecycle with version management."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    base_dir: str = Field(default=None, description="The base directory to use for the prompts")
    save_path: str = Field(default=None, description="The path to save the prompts")
    contract_path: str = Field(default=None, description="The path to save the prompt contract")
    
    def __init__(self, base_dir: Optional[str] = None, **kwargs):
        """Initialize the Prompt Manager."""
        super().__init__(**kwargs)
        self._registered_configs: Dict[str, PromptConfig] = {}  # prompt_name -> PromptConfig
    
    async def initialize(self, prompt_names: Optional[List[str]] = None):
        """Initialize prompts by names using prompt context manager with concurrent support.
        
        Args:
            prompt_names: List of prompt names to initialize. If None, initialize all registered prompts.
        """
        self.base_dir = assemble_project_path(os.path.join(config.workdir, "prompt"))
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "prompt.json")
        self.contract_path = os.path.join(self.base_dir, "contract.md")
        logger.info(f"| 📁 Prompt Manager base directory: {self.base_dir} with save path: {self.save_path} and contract path: {self.contract_path}")
        
        # Initialize prompt context manager
        self.prompt_context_manager = PromptContextManager(
            base_dir=self.base_dir,
            save_path=self.save_path,
            contract_path=self.contract_path,
        )
        await self.prompt_context_manager.initialize(prompt_names=prompt_names)
        
        logger.info("| ✅ Prompts initialization completed")
    
    async def register(self, prompt: Union[Prompt, Dict[str, Any]], *, override: bool = False, **kwargs: Any) -> PromptConfig:
        """Register a prompt class or template dictionary asynchronously.
        
        Args:
            prompt: Prompt class or template dictionary to register
            override: Whether to override existing registration
            **kwargs: Configuration for prompt initialization
            
        Returns:
            PromptConfig: Prompt configuration
        """
        prompt_config = await self.prompt_context_manager.register(prompt, override=override, **kwargs)
        self._registered_configs[prompt_config.name] = prompt_config
        return prompt_config
    
    async def list(self) -> List[str]:
        """List all registered prompts
        
        Returns:
            List[str]: List of prompt names
        """
        return await self.prompt_context_manager.list()
    
    async def get(self, prompt_name: str) -> Optional[Prompt]:
        """Get prompt instance by name
        
        Args:
            prompt_name: Prompt name
            
        Returns:
            Prompt: Prompt instance or None if not found
        """
        return await self.prompt_context_manager.get(prompt_name)
    
    async def get_info(self, prompt_name: str) -> Optional[PromptConfig]:
        """Get prompt configuration by name
        
        Args:
            prompt_name: Prompt name
            
        Returns:
            PromptConfig: Prompt configuration or None if not found
        """
        return await self.prompt_context_manager.get_info(prompt_name)
    
    async def cleanup(self):
        """Cleanup all prompts"""
        if hasattr(self, 'prompt_context_manager'):
            await self.prompt_context_manager.cleanup()
        self._registered_configs.clear()
    
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
        prompt_config = await self.prompt_context_manager.update(
            prompt_name, prompt, new_version=new_version, description=description, **kwargs
        )
        self._registered_configs[prompt_config.name] = prompt_config
        return prompt_config
    
    async def copy(self, prompt_name: str, new_name: Optional[str] = None,
                  new_version: Optional[str] = None, **override_config) -> PromptConfig:
        """Copy an existing prompt
        
        Args:
            prompt_name: Name of the prompt to copy
            new_name: New name for the copied prompt. If None, uses original name.
            new_version: New version for the copied prompt. If None, increments version.
            **override_config: Configuration overrides
            
        Returns:
            PromptConfig: New prompt configuration
        """
        prompt_config = await self.prompt_context_manager.copy(
            prompt_name, new_name, new_version, **override_config
        )
        self._registered_configs[prompt_config.name] = prompt_config
        return prompt_config
    
    async def unregister(self, prompt_name: str) -> bool:
        """Unregister a prompt
        
        Args:
            prompt_name: Name of the prompt to unregister
            
        Returns:
            True if unregistered successfully, False otherwise
        """
        success = await self.prompt_context_manager.unregister(prompt_name)
        if success and prompt_name in self._registered_configs:
            del self._registered_configs[prompt_name]
        return success
    
    async def restore(self, prompt_name: str, version: str, auto_initialize: bool = True) -> Optional[PromptConfig]:
        """Restore a specific version of a prompt from history
        
        Args:
            prompt_name: Name of the prompt
            version: Version string to restore
            auto_initialize: Whether to automatically initialize the restored prompt
            
        Returns:
            PromptConfig of the restored version, or None if not found
        """
        prompt_config = await self.prompt_context_manager.restore(prompt_name, version, auto_initialize)
        if prompt_config:
            self._registered_configs[prompt_config.name] = prompt_config
        return prompt_config
    
    async def get_system_message(self, 
                          prompt_name: Optional[str] = None,
                          modules: Dict[str, Any] = None, 
                          reload: bool = False, 
                          **kwargs):
        """Get a system message using SystemPrompt.
        
        Args:
            prompt_name: Name of the prompt (e.g., "tool_calling"). 
                        If None, will try to infer from kwargs or use default.
            modules: Modules to render in the template
            reload: Whether to reload the prompt
            **kwargs: Additional arguments (may include prompt_name for backward compatibility)
        """
        # Ensure prompt_manager is initialized
        if not hasattr(self, 'prompt_context_manager'):
            await self.initialize()
        
        return await self.prompt_context_manager.get_system_message(
            prompt_name=prompt_name,
            modules=modules,
            reload=reload,
            **kwargs
        )
    
    async def get_agent_message(self, 
                         prompt_name: Optional[str] = None,
                         modules: Dict[str, Any] = None, 
                         reload: bool = True, 
                         **kwargs):
        """Get an agent message using AgentMessagePrompt.
        
        Args:
            prompt_name: Name of the prompt (e.g., "tool_calling").
                        If None, will try to infer from kwargs or use default.
            modules: Modules to render in the template
            reload: Whether to reload the prompt
            **kwargs: Additional arguments (may include prompt_name for backward compatibility)
        """
        # Ensure prompt_manager is initialized
        if not hasattr(self, 'prompt_context_manager'):
            await self.initialize()
        
        return await self.prompt_context_manager.get_agent_message(
            prompt_name=prompt_name,
            modules=modules,
            reload=reload,
            **kwargs
        )
        
    async def get_messages(self,
        prompt_name: Optional[str] = None,
        system_modules: Dict[str, Any] = None,
        agent_modules: Dict[str, Any] = None,
        **kwargs
    ) -> List[Message]:
        """Get a system and agent message using SystemPrompt and AgentMessagePrompt.
        
        Args:
            prompt_name (str): Name of the prompt (e.g., "tool_calling_system_prompt").
            system_modules (Dict[str, Any]): Modules to render in the system prompt
            agent_modules (Dict[str, Any]): Modules to render in the agent message prompt
            **kwargs (Any): Additional arguments (may include prompt_name for backward compatibility)
            
        Returns:
            List[Message]: List of system and agent messages
        """
        return await self.prompt_context_manager.get_messages(
            prompt_name=prompt_name,
            system_modules=system_modules,
            agent_modules=agent_modules,
            **kwargs
        )
    
    async def get_variables(self, prompt_name: Optional[str] = None) -> Dict[str, Variable]:
        """Get all variables from a prompt.
        
        Args:
            prompt_name (str): Name of the prompt (e.g., "tool_calling").
            
        Returns:
            Dict[str, Variable]: Dictionary mapping variable names to Variable objects.
        """
        return await self.prompt_context_manager.get_variables(prompt_name=prompt_name)
    
    async def set_variables(self, prompt_name: str, variable_updates: Dict[str, Any], new_version: Optional[str] = None, description: Optional[str] = None) -> Dict[str, PromptConfig]:
        """Set variable values in a prompt and create a new version.
        
        Args:
            prompt_name: Name of the prompt to update (base name, e.g., "tool_calling")
            variable_updates: Dictionary mapping variable names to new values.
            new_version: New version string. If None, auto-increments from current version.
            description: Description for this version update    
            
        Returns:
            Dict[str, PromptConfig]: Dictionary mapping prompt names to updated configurations
                                     (e.g., {"tool_calling_system_prompt": PromptConfig, ...})
        """
        # The context manager returns a dict mapping prompt names to PromptConfig instances.
        updated_configs = await self.prompt_context_manager.set_variables(
            prompt_name=prompt_name,
            variable_updates=variable_updates,
            new_version=new_version,
            description=description,
        )
        # Update local registry for each returned PromptConfig
        for _, updated_config in updated_configs.items():
            self._registered_configs[updated_config.name] = updated_config
        return updated_configs
    
    async def get_trainable_variables(self, prompt_name: Optional[str] = None) -> Dict[str, Variable]:
        """Get all trainable variables from a prompt.
        
        Args:
            prompt_name (str): Name of the prompt (e.g., "tool_calling").
        """
        return await self.prompt_context_manager.get_trainable_variables(prompt_name=prompt_name)
    
    async def set_contract(self, prompt_names: Optional[List[str]] = None):
        """Set the contract for all prompts by aggregating their source code.

        Args:
            prompt_names: List of prompt names to include in the contract. If None, includes all registered prompts.
        """
        await self.prompt_context_manager.save_contract(prompt_names=prompt_names)

    async def get_contract(self) -> str:
        """Get the contract for all prompts

        Returns:
            str: Contract text content
        """
        return await self.prompt_context_manager.load_contract()


# Global Prompt Manager instance
prompt_manager = PromptManagerServer()
