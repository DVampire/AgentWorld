"""Browser Controller for managing browser automation through tools."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from inspect import cleandoc
from langchain.tools import BaseTool, StructuredTool

from src.tools.base import ToolResponse
from src.controller.base import BaseController
from src.registry import CONTROLLERS
from src.registry import ENVIRONMENTS


# Navigation Operations
_NAVIGATION_OPERATIONS_DESCRIPTION = """Navigation operations tool for browser navigation.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

Available operations:
1. goto: Navigate to a URL.
    - url: The URL to navigate to.
2. back: Go back to the previous page.
    - No parameters required.
3. forward: Go forward to the next page.
    - No parameters required.
4. refresh: Refresh the current page.
    - No parameters required.

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "goto", "url": "https://www.google.com"}
"""

# Interaction Operations  
_INTERACTION_OPERATIONS_DESCRIPTION = """Interaction operations tool for browser interactions.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

Available operations:
1. click: Click on an element.
    - index: The element index to click (required, must be > 0, found in browser_state).
    - while_holding_ctrl: Hold Ctrl while clicking (optional, opens in new tab).
2. type: Type text into an element.
    - index: The element index to type into (required, must exist in browser_state).
    - text: The text to type (required).
    - clear_existing: Clear existing text before typing (default: true).
3. scroll: Scroll the page.
    - direction: The direction to scroll ('up' or 'down', default: 'down').
    - num_pages: Number of pages to scroll (default: 1.0, can be fractional like 0.5).
4. send_keys: Send keyboard keys.
    - keys: The keys to send (e.g., 'Enter', 'Tab', 'Ctrl+A').
5. upload_file: Upload a file to an element.
    - index: The element index for file upload (must exist in browser_state).
    - path: The file path to upload.

IMPORTANT: All index values must exist in the current browser_state. Use data_extraction_operations to get available elements first.

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "click", "index": 123}
"""

# Data Extraction Operations
_DATA_EXTRACTION_OPERATIONS_DESCRIPTION = """Data extraction operations tool for getting page information.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

Available operations:
1. extract_structured_data: Extract structured data from the current page.
    - query: The query describing what data to extract (required).
    - extract_links: Whether to extract links (default: false).
2. get_dropdown_options: Get options from a dropdown element.
    - index: The element index of the dropdown (required).
3. select_dropdown_option: Select an option from a dropdown.
    - index: The element index of the dropdown (required).
    - text: The text of the option to select (required).
4. wait: Wait for a specified number of seconds.
    - seconds: Number of seconds to wait (default: 3).

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "extract_structured_data", "query": "product prices"}
"""

# Tab Management Operations
_TAB_MANAGEMENT_OPERATIONS_DESCRIPTION = """Tab management operations tool for managing browser tabs.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

Available operations:
1. open_tab: Open a new tab.
    - url: (optional) The URL to open in the new tab (default: 'about:blank').
2. close_tab: Close a tab.
    - tab_id: The ID of the tab to close.
3. switch_tab: Switch to a tab.
    - tab_id: The ID of the tab to switch to.
4. list_tabs: List all open tabs.
    - No parameters required.

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "open_tab", "url": "https://www.google.com"}
"""

# Pydantic models for each operation type
class NavigationOperationArgs(BaseModel):
    operation: str = Field(description="The navigation operation to execute")
    url: Optional[str] = Field(default=None, description="The URL to navigate to")

class InteractionOperationArgs(BaseModel):
    operation: str = Field(description="The interaction operation to execute")
    index: Optional[int] = Field(default=None, description="Element index (required for click, type, upload)")
    element_id: Optional[str] = Field(default=None, description="Element ID (alternative to index)")
    text: Optional[str] = Field(default=None, description="The text to type")
    direction: Optional[str] = Field(default="down", description="The direction to scroll")
    num_pages: Optional[float] = Field(default=1.0, description="Number of pages to scroll")
    keys: Optional[str] = Field(default=None, description="The keys to send")
    clear_existing: Optional[bool] = Field(default=True, description="Clear existing text before typing")
    while_holding_ctrl: Optional[bool] = Field(default=False, description="Hold Ctrl while clicking")
    path: Optional[str] = Field(default=None, description="File path for upload operations")

class DataExtractionOperationArgs(BaseModel):
    operation: str = Field(description="The data extraction operation to execute")
    file_path: Optional[str] = Field(default="screenshot.png", description="The path to save files")
    query: Optional[str] = Field(default=None, description="Query for structured data extraction")
    extract_links: Optional[bool] = Field(default=False, description="Extract links in structured data")
    index: Optional[int] = Field(default=None, description="Element index for dropdown operations")
    text: Optional[str] = Field(default=None, description="Text for dropdown option selection")
    seconds: Optional[int] = Field(default=3, description="Seconds to wait")

class TabManagementOperationArgs(BaseModel):
    operation: str = Field(description="The tab management operation to execute")
    tab_id: Optional[str] = Field(default=None, description="The ID of the tab")
    url: Optional[str] = Field(default="about:blank", description="The URL to open")


@CONTROLLERS.register_module(force=True)
class BrowserController(BaseController):
    def __init__(self, environment: Any, environment_rules: Any):
        # Build environment
        self.environment = self._build_environment(environment)
        self.environment_rules = environment_rules

        self._tools: Dict[str, BaseTool] = {}
        self._tool_configs: Dict[str, Dict[str, Any]] = {}
        
        self.state = None
        self.info = None
        self.done = None
        
    def _build_environment(self, environment: Any):
        environment = ENVIRONMENTS.build(environment)
        return environment
    
    async def get_state(self) -> str:
        """Get the state of the browser controller."""
        # Initialize environment if not done
        if not hasattr(self.environment, 'state') or self.environment.state is None:
            try:
                await self.environment.reset()
            except Exception as e:
                return f"<environment_browser_state>Error initializing browser: {str(e)}</environment_browser_state>"
        
        # Get current environment state
        try:
            current_url = await self.environment._get_current_url()
            tabs = await self.environment._get_tabs_info()
            state_info = f"URL: {current_url}, Tabs: {len(tabs)}"
        except Exception as e:
            state_info = f"Browser not ready: {str(e)}"
            
        state = cleandoc(f"""<environment_browser_state>
                         Current state: {state_info}
                         </environment_browser_state>""")
        return state

    async def initialize(self):
        """Initialize the browser controller."""
        self.state, self.info = await self.environment.reset()
        await self._register_tools()
    
    async def _navigation_operation_tool(self, **kwargs) -> ToolResponse:
        """Handle navigation operations through environment."""
        operation = kwargs.get('operation')
        
        try:
            # Create action for environment
            action = {
                "type": "navigation",
                "operation": operation,
                "params": {k: v for k, v in kwargs.items() if k != 'operation' and v is not None}
            }
            
            # Execute through environment
            state, reward, done, truncated, info = await self.environment.step(action)
            
            # Extract result from info
            action_result = info.get('action_result', {})
            result = action_result.get('content', 'No result available')
            success = action_result.get('success', False)
            
            # Update controller state
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=result)
                
        except Exception as e:
            return ToolResponse(content=f"Error in navigation operation '{operation}': {str(e)}")
    
    async def _interaction_operation_tool(self, **kwargs) -> ToolResponse:
        """Handle interaction operations through environment."""
        operation = kwargs.get('operation')
        
        try:
            # Create action for environment
            # 将element_id转换为index以兼容browser-use
            params = {k: v for k, v in kwargs.items() if k != 'operation' and v is not None}
            if 'element_id' in params and 'index' not in params:
                try:
                    params['index'] = int(params.pop('element_id'))
                except (ValueError, TypeError):
                    # 如果element_id不是数字，设置默认index
                    params['index'] = 1
            
            action = {
                "type": "interaction",
                "operation": operation,
                "params": params
            }
            
            # Execute through environment
            state, reward, done, truncated, info = await self.environment.step(action)
            
            # Extract result from info
            action_result = info.get('action_result', {})
            result = action_result.get('content', 'No result available')
            success = action_result.get('success', False)
            
            # Update controller state
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=result)
                
        except Exception as e:
            return ToolResponse(content=f"Error in interaction operation '{operation}': {str(e)}")
    
    async def _data_extraction_operation_tool(self, **kwargs) -> ToolResponse:
        """Handle data extraction operations through environment."""
        operation = kwargs.get('operation')
        
        try:
            # Create action for environment
            action = {
                "type": "data_extraction",
                "operation": operation,
                "params": {k: v for k, v in kwargs.items() if k != 'operation' and v is not None}
            }
            
            # Execute through environment
            state, reward, done, truncated, info = await self.environment.step(action)
            
            # Extract result from info
            action_result = info.get('action_result', {})
            result = action_result.get('content', 'No result available')
            success = action_result.get('success', False)
            
            # Update controller state
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=result)
                
        except Exception as e:
            return ToolResponse(content=f"Error in data extraction operation '{operation}': {str(e)}")
    
    async def _tab_management_operation_tool(self, **kwargs) -> ToolResponse:
        """Handle tab management operations through environment."""
        operation = kwargs.get('operation')
        
        try:
            # Create action for environment
            action = {
                "type": "tab_management",
                "operation": operation,
                "params": {k: v for k, v in kwargs.items() if k != 'operation' and v is not None}
            }
            
            # Execute through environment
            state, reward, done, truncated, info = await self.environment.step(action)
            
            # Extract result from info
            action_result = info.get('action_result', {})
            result = action_result.get('content', 'No result available')
            success = action_result.get('success', False)
            
            # Update controller state
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=result)
                
        except Exception as e:
            return ToolResponse(content=f"Error in tab management operation '{operation}': {str(e)}")
    
    async def _register_tools(self):
        """Register all browser tools."""
        
        # Navigation operations tool
        navigation_operation_tool = StructuredTool.from_function(
            name="navigation_operations",
            description=_NAVIGATION_OPERATIONS_DESCRIPTION,
            coroutine=self._navigation_operation_tool,
            args_schema=NavigationOperationArgs
        )
        self._tools["navigation_operations"] = navigation_operation_tool
        self._tool_configs["navigation_operations"] = {
            "name": navigation_operation_tool.name,
            "description": navigation_operation_tool.description,
            "args_schema": navigation_operation_tool.args_schema
        }
        
        # Interaction operations tool
        interaction_operation_tool = StructuredTool.from_function(
            name="interaction_operations",
            description=_INTERACTION_OPERATIONS_DESCRIPTION,
            coroutine=self._interaction_operation_tool,
            args_schema=InteractionOperationArgs
        )
        self._tools["interaction_operations"] = interaction_operation_tool
        self._tool_configs["interaction_operations"] = {
            "name": interaction_operation_tool.name,
            "description": interaction_operation_tool.description,
            "args_schema": interaction_operation_tool.args_schema
        }
        
        # Data extraction operations tool
        data_extraction_operation_tool = StructuredTool.from_function(
            name="data_extraction_operations",
            description=_DATA_EXTRACTION_OPERATIONS_DESCRIPTION,
            coroutine=self._data_extraction_operation_tool,
            args_schema=DataExtractionOperationArgs
        )
        self._tools["data_extraction_operations"] = data_extraction_operation_tool
        self._tool_configs["data_extraction_operations"] = {
            "name": data_extraction_operation_tool.name,
            "description": data_extraction_operation_tool.description,
            "args_schema": data_extraction_operation_tool.args_schema
        }
        
        # Tab management operations tool
        tab_management_operation_tool = StructuredTool.from_function(
            name="tab_management_operations",
            description=_TAB_MANAGEMENT_OPERATIONS_DESCRIPTION,
            coroutine=self._tab_management_operation_tool,
            args_schema=TabManagementOperationArgs
        )
        self._tools["tab_management_operations"] = tab_management_operation_tool
        self._tool_configs["tab_management_operations"] = {
            "name": tab_management_operation_tool.name,
            "description": tab_management_operation_tool.description,
            "args_schema": tab_management_operation_tool.args_schema
        }
        
    def list_tools(self) -> List[str]:
        """List all tools."""
        return list(self._tools.keys())
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tool_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool config by name."""
        return self._tool_configs.get(name)
    
    async def init_tools(self):
        """Initialize the tools."""
        await self.initialize()
