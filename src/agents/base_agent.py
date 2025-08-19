"""Base agent class for multi-agent system."""

from abc import ABC
from typing import Dict, List, Any, Optional, Union
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
import json

from src.prompts import prompt_manager
from src.models import model_manager
from src.tools import tool_manager
from src.logger import logger


class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""
    
    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        llm: Optional[BaseLanguageModel] = None,
        prompt_template: Optional[Union[str, PromptTemplate]] = None,
        prompt_name: Optional[str] = None,
        tools: Optional[List[Union[str, BaseTool]]] = None,
        **kwargs
    ):
        self.name = name
        self.model_name = model_name
        
        self.prompt_manager = prompt_manager
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        
        # Setup model
        self.model = self._setup_model(model_name, llm)
        
        # Setup prompt template
        self.prompt_template = self._setup_prompt_template(prompt_template, prompt_name)
        
        # Setup tools
        self.tools = []
        self._setup_tools(tools)
        
        self.state_graph = None
    
    def _setup_model(self, model_name: Optional[str], llm: Optional[BaseLanguageModel]):
        """Setup the language model."""
        if llm:
            # Use provided LLM directly
            return llm
        elif model_name:
            # Get model from ModelManager
            model = self.model_manager.get_model(model_name)
            if model:
                return model
            else:
                logger.warning(f"Warning: Model '{model_name}' not found in ModelManager")
        
        # Fallback to default model
        default_model = self.model_manager.get_model("gpt-4.1")
        if default_model:
            return default_model
        else:
            raise RuntimeError("No model available")
    
    def _setup_prompt_template(self, prompt_template: Optional[Union[str, PromptTemplate]], prompt_name: Optional[str]) -> PromptTemplate:
        """Setup the prompt template for the agent."""
        # If prompt_name is provided, try to get it from PromptManager
        if prompt_name:
            template = self.prompt_manager.get_template(prompt_name)
            if template:
                return template
            else:
                logger.warning(f"Warning: Prompt template '{prompt_name}' not found, using fallback")
        
        # If prompt_template is provided directly
        if isinstance(prompt_template, str):
            return PromptTemplate.from_template(prompt_template)
        elif isinstance(prompt_template, PromptTemplate):
            return prompt_template
        
        # Fallback to default template
        default_template = self.prompt_manager.get_template("default")
        if default_template:
            return default_template
        else:
            # Ultimate fallback
            return PromptTemplate.from_template(
                "You are {agent_name}, a helpful AI assistant.\n\n"
                "Current conversation:\n{chat_history}\n\n"
                "Human: {input}\n"
                "{agent_name}:"
            )
    
    def _setup_tools(self, tools: Optional[List[Union[str, BaseTool]]]):
        """Setup tools for the agent."""
        if not tools:
            return
        
        for tool in tools:
            if isinstance(tool, str):
                # Get tool by name from ToolManager
                tool_obj = self.tool_manager.get_tool(tool)
                if tool_obj:
                    self.tools.append(tool_obj)
                else:
                    logger.warning(f"Warning: Tool '{tool}' not found in ToolManager")
            elif isinstance(tool, BaseTool):
                # Add tool directly
                self.tools.append(tool)
    
    def get_prompt_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self.prompt_manager.get_template(name)
    
    def set_prompt_template(self, name: str):
        """Set the agent's prompt template by name."""
        template = self.prompt_manager.get_template(name)
        if template:
            self.prompt_template = template
        else:
            print(f"Warning: Prompt template '{name}' not found")
    
    def add_prompt_template(self, name: str, template: Union[str, Dict, PromptTemplate]):
        """Add a new prompt template to the manager."""
        self.prompt_manager.add_template(name, template)
    
    def list_available_prompts(self) -> List[str]:
        """List all available prompt templates."""
        return self.prompt_manager.list_templates()
    
    def get_model(self, name: str):
        """Get a model by name."""
        return self.model_manager.get_model(name)
    
    def set_model(self, name: str):
        """Set the agent's model by name."""
        model = self.model_manager.get_model(name)
        if model:
            self.model = model
        else:
            print(f"Warning: Model '{name}' not found")
    
    def list_available_models(self) -> List[str]:
        """List all available models."""
        return self.model_manager.list_models()
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tool_manager.get_tool(name)
    
    def add_tool(self, tool: Union[str, BaseTool]):
        """Add a tool to the agent."""
        if isinstance(tool, str):
            tool_obj = self.tool_manager.get_tool(tool)
            if tool_obj:
                self.tools.append(tool_obj)
            else:
                print(f"Warning: Tool '{tool}' not found in ToolManager")
        elif isinstance(tool, BaseTool):
            self.tools.append(tool)
    
    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent."""
        self.tools = [tool for tool in self.tools if tool.name != tool_name]
    
    def list_available_tools(self) -> List[str]:
        """List all available tools."""
        return self.tool_manager.list_tools()
    
    def get_tools(self) -> List[BaseTool]:
        """Get all tools available to this agent."""
        return self.tools
    
    def get_prompt_variables(self) -> Dict[str, Any]:
        """Get variables for prompt template."""
        # Get tool descriptions
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
        return {
            "agent_name": self.name,
            "tools": "\n".join(tool_descriptions) if tool_descriptions else "No tools available",
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the agent."""
        return {
            "name": self.name,
            "model": self.model_manager.get_model_info(self.model_name),
            "prompt_template": self.prompt_template.template,
            "tools": [tool.name for tool in self.tools],
            "available_prompts": len(self.list_available_prompts()),
            "available_models": len(self.list_available_models()),
            "available_tools": len(self.list_available_tools())
        }
        
    def __str__(self):
        return json.dumps(self.get_agent_info(), indent=4)
    
    def __repr__(self):
        return self.__str__()
