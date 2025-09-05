from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from inspect import cleandoc
from langchain.tools import BaseTool, StructuredTool

from src.tools.base import ToolResponse
from src.controller.base import BaseController
from src.registry import CONTROLLERS
from src.registry import ENVIRONMENTS

# File Operations
_FILE_OPERATIONS_DESCRIPTION = """File operations tool for managing individual files.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

Available operations:
1. read: Read a file from the file system.
    - file_path: The absolute path of the file to read.
    - start_line: (optional) Start line number for reading a range.
    - end_line: (optional) End line number for reading a range.
2. write: Write content to a file.
    - file_path: The absolute path of the file to write.
    - content: The content to write to the file.
    - mode: (optional) Write mode, 'w' for overwrite (default) or 'a' for append.
3. replace: Replace a string in a file.
    - file_path: The absolute path of the file to modify.
    - old_string: The string to replace.
    - new_string: The new string to replace with.
    - start_line: (optional) Start line number for range replacement.
    - end_line: (optional) End line number for range replacement.
4. delete: Delete a file.
    - file_path: The absolute path of the file to delete.
5. copy: Copy a file from source to destination.
    - src_path: The absolute path of the source file.
    - dst_path: The absolute path of the destination file.
6. move: Move a file from source to destination.
    - src_path: The absolute path of the source file.
    - dst_path: The absolute path of the destination file.
7. rename: Rename a file or directory.
    - old_path: The absolute path of the file/directory to rename.
    - new_path: The absolute path of the new name.
8. get_info: Get detailed information about a file.
    - file_path: The absolute path of the file.
    - include_stats: (optional) Whether to include file statistics (default: True).

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "read", "file_path": "/path/to/file.txt"}
"""

# Directory Operations
_DIRECTORY_OPERATIONS_DESCRIPTION = """Directory operations tool for managing directories.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

Available operations:
1. create_dir: Create a directory.
    - dir_path: The absolute path of the directory to create.
2. delete_dir: Delete a directory.
    - dir_path: The absolute path of the directory to delete.
3. tree: Show directory tree structure.
    - dir_path: The absolute path of the directory to show.
    - max_depth: (optional) Maximum depth to show (default: 3).
    - show_hidden: (optional) Whether to show hidden files (default: False).
    - exclude_patterns: (optional) List of patterns to exclude.
    - file_types: (optional) List of file extensions to include.
4. describe: Describe the file system with directory structure and file information.
    - No parameters required.

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "create_dir", "dir_path": "/path/to/directory"}
"""

# Search Operations
_SEARCH_OPERATIONS_DESCRIPTION = """Search operations tool for finding files and content.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

Available operations:
1. search: Search for files by name or content.
    - search_path: The absolute path of the directory to search in, or file path for single file search.
    - query: The search query string.
    - search_type: Search type, 'name' for filename search or 'content' for content search (default: 'name').
    - file_types: (optional) List of file extensions to filter by.
    - case_sensitive: (optional) Whether search is case sensitive (default: False).
    - max_results: (optional) Maximum number of results to return (default: 50).

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "search", "search_path": "/path/to/directory", "query": "search query"}
"""

# Permission Operations
_PERMISSION_OPERATIONS_DESCRIPTION = """Permission operations tool for managing file and directory permissions.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

Available operations:
1. change_permissions: Change file or directory permissions.
    - file_path: The absolute path of the file or directory.
    - permissions: The new permissions in octal format (e.g., '755', '644').

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "change_permissions", "file_path": "/path/to/file", "permissions": "755"}
"""

# Pydantic models for each operation type
class FileOperationArgs(BaseModel):
    operation: str = Field(description="The file operation to execute")
    file_path: Optional[str] = Field(default=None, description="The absolute path of the file")
    src_path: Optional[str] = Field(default=None, description="The absolute path of the source file")
    dst_path: Optional[str] = Field(default=None, description="The absolute path of the destination file")
    old_path: Optional[str] = Field(default=None, description="The absolute path of the file to rename")
    new_path: Optional[str] = Field(default=None, description="The absolute path of the new name")
    content: Optional[str] = Field(default=None, description="The content to write or append")
    old_string: Optional[str] = Field(default=None, description="The string to replace")
    new_string: Optional[str] = Field(default=None, description="The new string to replace with")
    mode: Optional[str] = Field(default="w", description="Write mode: 'w' for overwrite, 'a' for append")
    start_line: Optional[int] = Field(default=None, description="Start line number for range operations")
    end_line: Optional[int] = Field(default=None, description="End line number for range operations")
    include_stats: Optional[bool] = Field(default=True, description="Whether to include file statistics")

class DirectoryOperationArgs(BaseModel):
    operation: str = Field(description="The directory operation to execute")
    dir_path: Optional[str] = Field(default=None, description="The absolute path of the directory")
    max_depth: Optional[int] = Field(default=3, description="Maximum depth for tree display")
    show_hidden: Optional[bool] = Field(default=False, description="Whether to show hidden files")
    exclude_patterns: Optional[List[str]] = Field(default=None, description="List of patterns to exclude")
    file_types: Optional[List[str]] = Field(default=None, description="List of file extensions to include")

class SearchOperationArgs(BaseModel):
    operation: str = Field(description="The search operation to execute")
    search_path: str = Field(description="The absolute path to search in")
    query: str = Field(description="The search query string")
    search_type: Optional[str] = Field(default="name", description="Search type: 'name' or 'content'")
    file_types: Optional[List[str]] = Field(default=None, description="List of file extensions to filter by")
    case_sensitive: Optional[bool] = Field(default=False, description="Whether search is case sensitive")
    max_results: Optional[int] = Field(default=50, description="Maximum number of results to return")

class PermissionOperationArgs(BaseModel):
    operation: str = Field(description="The permission operation to execute")
    file_path: str = Field(description="The absolute path of the file or directory")
    permissions: str = Field(description="The new permissions in octal format (e.g., '755', '644')")

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
    
    async def get_state(self) -> str:
        """Get the state of the file system controller."""
        state = cleandoc(f"""<environment_file_system_state>
                         Current state: {self.state['prompt']}
                         </environment_file_system_state>""")
        return state

    async def initialize(self):
        """Initialize the file system controller."""
        self.state, self.info = await self.environment.reset()
        await self._register_tools()
    
    async def _file_operation_tool(self, **kwargs) -> ToolResponse:
        """Handle file operations through environment."""
        operation = kwargs.get('operation')
        
        try:
            # Create action for environment
            action = {
                "type": "file_operations",
                "operation": operation,
                "params": {k: v for k, v in kwargs.items() if k != 'operation' and v is not None}
            }
            
            # Execute through environment
            state, reward, done, truncated, info = await self.environment.step(action)
            
            # Extract result from info
            result = info.get('result', 'No result available')
            success = info.get('success', False)
            
            # Update controller state
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=result)
                
        except Exception as e:
            return ToolResponse(content=f"Error in file operation '{operation}': {str(e)}")
    
    async def _directory_operation_tool(self, **kwargs) -> ToolResponse:
        """Handle directory operations through environment."""
        operation = kwargs.get('operation')
        
        try:
            # Create action for environment
            action = {
                "type": "directory_operations",
                "operation": operation,
                "params": {k: v for k, v in kwargs.items() if k != 'operation' and v is not None}
            }
            
            # Execute through environment
            state, reward, done, truncated, info = await self.environment.step(action)
            
            # Extract result from info
            result = info.get('result', 'No result available')
            success = info.get('success', False)
            
            # Update controller state
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=result)
                
        except Exception as e:
            return ToolResponse(content=f"Error in directory operation '{operation}': {str(e)}")
    
    async def _search_operation_tool(self, **kwargs) -> ToolResponse:
        """Handle search operations through environment."""
        operation = kwargs.get('operation')
        
        try:
            # Create action for environment
            action = {
                "type": "search_operations",
                "operation": operation,
                "params": {k: v for k, v in kwargs.items() if k != 'operation' and v is not None}
            }
            
            # Execute through environment
            state, reward, done, truncated, info = await self.environment.step(action)
            
            # Extract result from info
            result = info.get('result', 'No result available')
            success = info.get('success', False)
            
            # Update controller state
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=result)
                
        except Exception as e:
            return ToolResponse(content=f"Error in search operation '{operation}': {str(e)}")
    
    async def _permission_operation_tool(self, **kwargs) -> ToolResponse:
        """Handle permission operations through environment."""
        operation = kwargs.get('operation')
        
        try:
            # Create action for environment
            action = {
                "type": "permission_operations",
                "operation": operation,
                "params": {k: v for k, v in kwargs.items() if k != 'operation' and v is not None}
            }
            
            # Execute through environment
            state, reward, done, truncated, info = await self.environment.step(action)
            
            # Extract result from info
            result = info.get('result', 'No result available')
            success = info.get('success', False)
            
            # Update controller state
            self.state = state
            self.info = info
            self.done = done
            
            return ToolResponse(content=result)
                
        except Exception as e:
            return ToolResponse(content=f"Error in permission operation '{operation}': {str(e)}")
    
    async def _register_tools(self):
        """Register all file system tools."""
        
        # File operations tool
        file_operation_tool = StructuredTool.from_function(
            name="file_operations",
            description=_FILE_OPERATIONS_DESCRIPTION,
            coroutine=self._file_operation_tool,
            args_schema=FileOperationArgs
        )
        self._tools["file_operations"] = file_operation_tool
        self._tool_configs["file_operations"] = {
            "name": file_operation_tool.name,
            "description": file_operation_tool.description,
            "args_schema": file_operation_tool.args_schema
        }
        
        # Directory operations tool
        directory_operation_tool = StructuredTool.from_function(
            name="directory_operations",
            description=_DIRECTORY_OPERATIONS_DESCRIPTION,
            coroutine=self._directory_operation_tool,
            args_schema=DirectoryOperationArgs
        )
        self._tools["directory_operations"] = directory_operation_tool
        self._tool_configs["directory_operations"] = {
            "name": directory_operation_tool.name,
            "description": directory_operation_tool.description,
            "args_schema": directory_operation_tool.args_schema
        }
        
        # Search operations tool
        search_operation_tool = StructuredTool.from_function(
            name="search_operations",
            description=_SEARCH_OPERATIONS_DESCRIPTION,
            coroutine=self._search_operation_tool,
            args_schema=SearchOperationArgs
        )
        self._tools["search_operations"] = search_operation_tool
        self._tool_configs["search_operations"] = {
            "name": search_operation_tool.name,
            "description": search_operation_tool.description,
            "args_schema": search_operation_tool.args_schema
        }
        
        # Permission operations tool
        permission_operation_tool = StructuredTool.from_function(
            name="permission_operations",
            description=_PERMISSION_OPERATIONS_DESCRIPTION,
            coroutine=self._permission_operation_tool,
            args_schema=PermissionOperationArgs
        )
        self._tools["permission_operations"] = permission_operation_tool
        self._tool_configs["permission_operations"] = {
            "name": permission_operation_tool.name,
            "description": permission_operation_tool.description,
            "args_schema": permission_operation_tool.args_schema
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
