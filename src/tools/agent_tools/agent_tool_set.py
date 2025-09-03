"""Agent tool set for managing workflow and agent-specific tools."""
from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool

from src.tools.agent_tools.browser import BrowserTool
from src.tools.agent_tools.deep_researcher import DeepResearcherTool
from src.config import config


class AgentToolSet:
    """Agent tool set containing workflow and agent-specific tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self):
        """Initialize the tool set by loading all agent tools asynchronously."""
        await self._load_agent_tools()
    
    async def _load_agent_tools(self):
        """Load all agent tools asynchronously."""
        # Load browser tool
        browser_tool = BrowserTool(
            model_name=config.browser_tool.model_name,
        )
        self._tools["browser"] = browser_tool
        self._tool_configs["browser"] = browser_tool.get_tool_config()
        
        # Load deep researcher tool
        deep_researcher_tool = DeepResearcherTool(
            model_name=config.deep_researcher_tool.model_name,
        )
        self._tools["deep_researcher"] = deep_researcher_tool
        self._tool_configs["deep_researcher"] = deep_researcher_tool.get_tool_config()

    def list_tools(self) -> List[str]:
        """Get all agent tools."""
        return list(self._tools.keys())
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a specific tool by name."""
        return self._tools.get(name)
    
    def get_tool_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get tool configuration by name."""
        return self._tool_configs.get(name)
    
    def list_tool_names(self) -> List[str]:
        """List all available tool names."""
        return list(self._tools.keys())
    
    def list_tools_by_category(self, category: str) -> List[BaseTool]:
        """List tools by category."""
        category_tools = []
        for name, config in self._tool_configs.items():
            if config.get("category") == category:
                tool = self._tools.get(name)
                if tool:
                    category_tools.append(tool)
        return category_tools
    
    def list_categories(self) -> List[str]:
        """List all available tool categories."""
        categories = set()
        for config in self._tool_configs.values():
            if "category" in config:
                categories.add(config["category"])
        return list(categories)
    
    def get_tool_summary(self) -> Dict[str, Any]:
        """Get a summary of all tools and their configurations."""
        return {
            "total_tools": len(self._tools),
            "tools": self._tool_configs,
            "categories": self.list_categories()
        }
        
    async def init_tools(self):
        """Factory method to create and initialize an AgentToolSet asynchronously."""
        await self.initialize()
