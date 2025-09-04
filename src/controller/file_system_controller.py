from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from inspect import cleandoc
from langchain.tools import Tool, BaseTool

from src.tools.base import ToolResponse
from src.controller.base import BaseController
from src.registry import CONTROLLERS
from src.registry import ENVIRONMENTS

_FILE_SYSTEM_ACTION_DESCRIPTION = """File system tool for file system environment.
Use this tool to manage files in the file system environment.

Available operations:
1. RESET: Reset the file system environment.
2. READ: Read a file from the file system.
    - filename: The name of the file to read.
3. WRITE: Write content to a file.
    - filename: The name of the file to write.
    - content: The content to write to the file.
4. APPEND: Append content to a file.
    - filename: The name of the file to append to.
    - content: The content to append to the file.
5. REPLACE: Replace a string in a file.
    - filename: The name of the file to modify.
    - old_string: The string to replace.
    - new_string: The new string to replace with.

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "reset"}
Example: {"operation": "read", "filename": "example.txt"}
Example: {"operation": "write", "filename": "example.txt", "content": "Hello World"}
Example: {"operation": "append", "filename": "example.txt", "content": "More content"}
Example: {"operation": "replace", "filename": "example.txt", "old_string": "Hello", "new_string": "Hi"}
"""

class FileSystemActionArgs(BaseModel):
    operation: str = Field(description="The operation to execute")
    filename: Optional[str] = Field(
        default=None,
        description="The name of the file to operate on"
    )
    content: Optional[str] = Field(
        default=None,
        description="The content to write or append to the file"
    )
    old_string: Optional[str] = Field(
        default=None,
        description="The string to replace in the file"
    )
    new_string: Optional[str] = Field(
        default=None,
        description="The new string to replace with"
    )

@CONTROLLERS.register_module(force=True)
class FileSystemController(BaseController):
    def __init__(self, environment: Any, environment_rules: Any):
        # build environment
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

    async def initialize(self):
        """Initialize the file system controller."""
        await self._register_tools()
    
    async def _action_tool(
        self, 
        operation: str, 
        filename: Optional[str] = None,
        content: Optional[str] = None,
        old_string: Optional[str] = None,
        new_string: Optional[str] = None
    ) -> ToolResponse:
        
        if operation == "reset":
            try:
                state, info = self.environment.reset()
            except Exception as e:
                return ToolResponse(content=f"Error in resetting the file system environment: {str(e)}")
            
            done = info["done"]
            
            state_description = cleandoc(f"""Reset the file system environment successfully.
            The state is:
            {state['description']}
            
            Todo contents:
            {state.get('todo_contents', 'No todo contents')}
            
            The environment is {'done' if done else 'not done'}.
            """)
            
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=state_description)
        
        elif operation == "read":
            if not filename:
                return ToolResponse(content="Error: filename is required for read operation")
            
            try:
                state, reward, done, truncated, info = self.environment.step(
                    operation="read",
                    filename=filename
                )
            except Exception as e:
                return ToolResponse(content=f"Error in reading file: {str(e)}")
            
            state_description = cleandoc(f"""Read file '{filename}' successfully.
            The state is:
            {state['description']}
            
            Todo contents:
            {state.get('todo_contents', 'No todo contents')}
            
            Operation result:
            {state.get('state', 'No operation result')}
            
            The environment is {'done' if done else 'not done'}.
            """)
            
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=state_description)
        
        elif operation == "write":
            if not filename:
                return ToolResponse(content="Error: filename is required for write operation")
            if content is None:
                return ToolResponse(content="Error: content is required for write operation")
            
            try:
                state, reward, done, truncated, info = self.environment.step(
                    operation="write",
                    filename=filename,
                    content=content
                )
            except Exception as e:
                return ToolResponse(content=f"Error in writing file: {str(e)}")
            
            state_description = cleandoc(f"""Write to file '{filename}' successfully.
            The state is:
            {state['description']}
            
            Todo contents:
            {state.get('todo_contents', 'No todo contents')}
            
            Operation result:
            {state.get('state', 'No operation result')}
            
            The environment is {'done' if done else 'not done'}.
            """)
            
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=state_description)
        
        elif operation == "append":
            if not filename:
                return ToolResponse(content="Error: filename is required for append operation")
            if content is None:
                return ToolResponse(content="Error: content is required for append operation")
            
            try:
                state, reward, done, truncated, info = self.environment.step(
                    operation="append",
                    filename=filename,
                    content=content
                )
            except Exception as e:
                return ToolResponse(content=f"Error in appending to file: {str(e)}")
            
            state_description = cleandoc(f"""Append to file '{filename}' successfully.
            The state is:
            {state['description']}
            
            Todo contents:
            {state.get('todo_contents', 'No todo contents')}
            
            Operation result:
            {state.get('state', 'No operation result')}
            
            The environment is {'done' if done else 'not done'}.
            """)
            
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=state_description)
        
        elif operation == "replace":
            if not filename:
                return ToolResponse(content="Error: filename is required for replace operation")
            if old_string is None:
                return ToolResponse(content="Error: old_string is required for replace operation")
            if new_string is None:
                return ToolResponse(content="Error: new_string is required for replace operation")
            
            try:
                state, reward, done, truncated, info = self.environment.step(
                    operation="replace",
                    filename=filename,
                    old_string=old_string,
                    new_string=new_string
                )
            except Exception as e:
                return ToolResponse(content=f"Error in replacing string in file: {str(e)}")
            
            state_description = cleandoc(f"""Replace string in file '{filename}' successfully.
            The state is:
            {state['description']}
            
            Todo contents:
            {state.get('todo_contents', 'No todo contents')}
            
            Operation result:
            {state.get('state', 'No operation result')}
            
            The environment is {'done' if done else 'not done'}.
            """)
            
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=state_description)
        else:
            return ToolResponse(content=f"Invalid operation: {operation}")
    
    async def _register_tools(self):
        
        # register action tool
        action_tool = Tool(
            name="file_system_action",
            description=_FILE_SYSTEM_ACTION_DESCRIPTION,
            func=self._action_tool,
            args_schema=FileSystemActionArgs
        )
        action_tool_config = {
            "name": action_tool.name,
            "description": action_tool.description,
            "args_schema": action_tool.args_schema
        }
        self._tools["file_system_action"] = action_tool
        self._tool_configs["file_system_action"] = action_tool_config
        
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
