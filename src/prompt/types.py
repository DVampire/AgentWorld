"""Prompt Context Protocol (PCP) Types

Core type definitions for the Prompt Context Protocol.
"""
from typing import Any, Dict, Optional, Type, Literal, List, Union, TYPE_CHECKING, Union
from pydantic import BaseModel, Field, ConfigDict

from src.logger import logger
from src.message import Message, SystemMessage, HumanMessage, ContentPartText
from src.optimizer.types import Variable
from src.dynamic import dynamic_manager

class Prompt(BaseModel):
    """Base class for all prompt templates with rendering capabilities.
    
    This class serves two purposes:
    1. Registration: Subclasses define system_prompt and agent_message_prompt properties
    2. Instance: When initialized with prompt_config or prompt_dict, provides rendering methods
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(description="The type of the prompt, e.g. 'system_prompt' or 'agent_message_prompt'")
    name: str = Field(description="The name of the prompt")
    description: str = Field(description="The description of the prompt")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the prompt")
    prompt_config: Optional[Dict[str, Any]] = Field(default=None, description="The prompt information")
    
    prompt_variable: Optional[Variable] = Field(default=None, description="The prompt variable")
    message: Optional[Message] = Field(default=None, description="The message")
    
    def __init__(self, prompt_config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize Prompt with prompt configuration."""
        super().__init__(**kwargs)
        self.prompt_config = prompt_config if prompt_config is not None else self.prompt_config
        
    async def initialize(self) -> None:
        """Initialize the prompt."""
        pass
    
    async def _load_prompt_variable(self) -> None:
        """Load prompt template asynchronously."""
        if self.prompt_variable is not None:
            return
        
        if self.prompt_config is None:
            raise ValueError("Cannot load prompt: prompt_config is None")
        
        try:
            self.prompt_variable = Variable.from_dict(self.prompt_config)
        except Exception as e:
            raise RuntimeError(f"Failed to load prompt: {e}")
    
    async def get_variable(self, reload: bool = False) -> Variable:
        """Get the prompt Variable instance.
        
        Args:
            reload: Whether to reload the prompt
            
        Returns:
            Variable: The prompt Variable instance
        """
        if self.prompt_variable is None or reload:
            await self._load_prompt_variable()
        return self.prompt_variable
    
    async def get_trainable_variable(self) -> List[Variable]:
        """Get all trainable variables from the prompt.
        
        Returns:
            List[Variable]: List of all trainable variables from the prompt
        """
        if self.prompt_variable is None:
            await self._load_prompt_variable()
        return self.prompt_variable.get_trainable_variables()
    
    async def get_message(
        self,
        modules: Optional[Dict[str, Any]] = None,
        reload: bool = False,
        **kwargs
    ):
        """Get the rendered message.
        
        Args:
            modules: Modules to render in the template
            reload: Whether to reload the prompt
            **kwargs: Additional arguments
            
        Returns:
            SystemMessage or HumanMessage depending on prompt type
        """
        # Load prompt if not already loaded or if reloading
        if self.prompt_variable is None or reload:
            await self._load_prompt_variable()
        
        is_system_prompt = self.type == "system_prompt"
        
        # Cache check (only for system prompts)
        if is_system_prompt and not reload and self.message is not None:
            return self.message
        
        try:
            # Build modules from variable tree if not provided
            if modules is None or len(modules) == 0:
                modules = self.prompt_variable.get_modules()
            else:
                # Merge provided modules with variable tree modules
                variable_modules = self.prompt_variable.get_modules()
                modules = {**variable_modules, **modules}
            
            prompt_str = self.prompt_variable.render(modules)
            
            # Return appropriate message type based on prompt type
            if is_system_prompt:
                self.message = SystemMessage(content=prompt_str)
            else:
                # Agent message prompt
                contents = [
                    ContentPartText(text=prompt_str),
                ]
                self.message = HumanMessage(content=contents)
            
            return self.message
            
        except Exception as e:
            logger.warning(f"Failed to render prompt: {e}")
            raise RuntimeError(f"Failed to render prompt: {e}")

    def __str__(self):
        return f"Prompt(name={self.name}, description={self.description})"

    def __repr__(self):
        return self.__str__()


class PromptConfig(BaseModel):
    """Prompt configuration for registration"""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the prompt")
    type: str = Field(description="The type of the prompt")
    description: str = Field(description="The description of the prompt")
    version: str = Field(default="1.0.0", description="Version of the prompt")
    template: str = Field(description="The template string for the prompt")
    variables: Optional[Union[Dict[str, 'Variable'], 'Variable']] = Field(default=None, description="The variables used in the template. Can be Dict[str, Variable] or single Variable")
    cls: Optional[Type[Prompt]] = Field(default=None, description="The class of the prompt")
    instance: Optional[Any] = Field(default=None, description="The instance of the prompt")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The initialization configuration of the prompt")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the prompt")
    code: Optional[str] = Field(default=None, description="Source code for dynamically generated prompt classes (used when cls cannot be imported from a module)")
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Dump the model to a dictionary, recursively serializing nested Pydantic models."""
        def serialize_variables(vars_data: Any) -> Any:
            """Recursively serialize Variable objects to dictionaries."""
            if vars_data is None:
                return None
            elif isinstance(vars_data, Variable):
                # Convert Variable to dict, excluding non-serializable fields
                return {
                    "name": vars_data.name,
                    "type": vars_data.type,
                    "description": vars_data.description,
                    "require_grad": vars_data.require_grad,
                    "template": vars_data.template,
                    "variables": serialize_variables(vars_data.variables),
                }
            elif isinstance(vars_data, dict):
                # Recursively process dictionary values
                return {k: serialize_variables(v) for k, v in vars_data.items()}
            elif isinstance(vars_data, (list, tuple)):
                # Recursively process list items
                return [serialize_variables(item) for item in vars_data]
            else:
                # Primitive types (str, int, etc.) - return as-is
                return vars_data
        
        result = {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "version": self.version,
            "template": self.template,
            "variables": serialize_variables(self.variables),
            "metadata": self.metadata,
            "config": self.config,
            "cls": dynamic_manager.get_class_string(self.cls) if self.cls else None,
            "instance": None,
            "code": self.code,
        }
        return result
    
    @classmethod
    def model_validate(cls, data: Dict[str, Any]) -> 'PromptConfig':
        """Validate the model from a dictionary."""
        name = data.get("name")
        prompt_type = data.get("type")
        description = data.get("description")
        version = data.get("version", "1.0.0")
        template = data.get("template", "")
        variables = data.get("variables")
        metadata = data.get("metadata", {})
        config_dict = data.get("config", {})
        
        cls_ = None
        code = data.get("code")
        if code:
            class_name = dynamic_manager.extract_class_name_from_code(code)
            if class_name:
                try:
                    cls_ = dynamic_manager.load_class(
                        code,
                        class_name=class_name,
                        base_class=Prompt,
                        context="prompt"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load prompt class from code: {e}")
                    cls_ = None
            else:
                cls_ = None
        else:
            cls_ = None
        
        instance = data.get("instance", None)
        
        return cls(
            name=name,
            type=prompt_type,
            description=description,
            version=version,
            template=template,
            variables=variables,
            cls=cls_,
            instance=instance,
            config=config_dict,
            metadata=metadata,
            code=code
        )
    
    def __str__(self):
        return f"PromptConfig(name={self.name}, type={self.type}, description={self.description}, version={self.version})"
    
    def __repr__(self):
        return self.__str__()
