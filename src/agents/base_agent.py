"""Base agent class for multi-agent system."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json
import asyncio

from src.utils import Singleton
from src.agents.prompts import PromptManager
from src.models import ModelManager
from src.tools import ToolManager


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
        mcp_tools: Optional[List[Dict]] = None,
        **kwargs
    ):
        self.name = name
        self.prompt_manager = PromptManager()
        self.model_manager = ModelManager()
        self.tool_manager = ToolManager()
        
        # Setup model
        self.model = self._setup_model(model_name, llm)
        
        # Setup prompt template
        self.prompt_template = self._setup_prompt_template(prompt_template, prompt_name)
        
        # Setup tools
        self.tools = []
        self.mcp_tools = mcp_tools or []
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
                return model.llm
            else:
                print(f"Warning: Model '{model_name}' not found in ModelManager")
        
        # Fallback to default model
        default_model = self.model_manager.get_model("gpt-4")
        if default_model:
            return default_model.llm
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
                print(f"Warning: Prompt template '{prompt_name}' not found, using fallback")
        
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
                    print(f"Warning: Tool '{tool}' not found in ToolManager")
            elif isinstance(tool, BaseTool):
                # Add tool directly
                self.tools.append(tool)
        
        # Add MCP tools if provided
        if self.mcp_tools:
            for mcp_tool in self.mcp_tools:
                # Convert MCP tool to LangChain tool
                langchain_tool = self._create_mcp_tool(mcp_tool)
                if langchain_tool:
                    self.tools.append(langchain_tool)
    
    def _create_mcp_tool(self, mcp_tool: Dict) -> Optional[BaseTool]:
        """Convert MCP tool definition to LangChain tool."""
        try:
            from langchain.tools import tool
            
            @tool
            async def mcp_tool_wrapper(**kwargs):
                """MCP tool wrapper."""
                # Here you would implement the actual MCP tool call
                # For now, we'll return a placeholder
                return f"MCP tool {mcp_tool.get('name', 'unknown')} called with {kwargs}"
            
            # Set tool metadata
            mcp_tool_wrapper.name = mcp_tool.get('name', 'mcp_tool')
            mcp_tool_wrapper.description = mcp_tool.get('description', 'MCP tool')
            
            return mcp_tool_wrapper
        except Exception as e:
            print(f"Failed to create MCP tool: {e}")
            return None
    
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
            self.model = model.llm
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
    
    @abstractmethod
    async def create_graph(self) -> StateGraph:
        """Create the agent's state graph."""
        pass
    
    @abstractmethod
    async def process_message(self, message: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message and return updated state."""
        pass
    
    def get_tools(self) -> List[BaseTool]:
        """Get all tools available to this agent."""
        return self.tools
    
    def add_mcp_tool(self, mcp_tool: Dict):
        """Add an MCP tool to the agent."""
        langchain_tool = self._create_mcp_tool(mcp_tool)
        if langchain_tool:
            self.tools.append(langchain_tool)
            self.mcp_tools.append(mcp_tool)
    
    def get_prompt_variables(self) -> Dict[str, Any]:
        """Get variables for prompt template."""
        # Get tool descriptions
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
        # Get MCP tool descriptions
        mcp_tool_descriptions = []
        for mcp_tool in self.mcp_tools:
            mcp_tool_descriptions.append(f"- {mcp_tool.get('name', 'unknown')}: {mcp_tool.get('description', 'MCP tool')}")
        
        return {
            "agent_name": self.name,
            "tools": "\n".join(tool_descriptions) if tool_descriptions else "No tools available",
            "mcp_tools": "\n".join(mcp_tool_descriptions) if mcp_tool_descriptions else "No MCP tools available",
            "file_tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools if "file" in tool.name.lower()]),
            "web_tools": "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools if "web" in tool.name.lower() or "search" in tool.name.lower()])
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the agent."""
        return {
            "name": self.name,
            "model": self.model_manager.get_model_info("gpt-4") if self.model else None,
            "prompt_template": self.prompt_template.template if self.prompt_template else None,
            "tools": [tool.name for tool in self.tools],
            "mcp_tools": [tool.get('name') for tool in self.mcp_tools],
            "available_prompts": len(self.list_available_prompts()),
            "available_models": len(self.list_available_models()),
            "available_tools": len(self.list_available_tools())
        }
