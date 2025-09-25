"""Tool Manager Workflow - Automatic tool discovery and registration to TCP."""

import asyncio
import os
import importlib
import inspect
from typing import List, Dict, Any, Optional, Type, Union
from pathlib import Path
from pydantic import BaseModel, Field

from src.tools.protocol.tool import BaseTool, WrappedTool
from src.tools.protocol.types import ToolInfo, ToolResponse
from src.tools.protocol.server import tcp
from src.logger import logger
from src.config import config
from src.infrastructures.models import model_manager


class ToolDiscoveryResult(BaseModel):
    """Result of tool discovery operation."""
    tool_name: str
    tool_class: Type[BaseTool]
    module_path: str
    is_valid: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}


class ToolRegistrationResult(BaseModel):
    """Result of tool registration operation."""
    tool_name: str
    success: bool
    error_message: Optional[str] = None
    was_already_registered: bool = False


class ToolManagerArgs(BaseModel):
    """Arguments for the tool manager TCP tool."""
    operation: str = Field(description="Operation to perform: discover, register, analyze, summary, cleanup")
    directory_path: Optional[str] = Field(
        default=None,
        description="Directory path to scan for tools (required for discover operation)"
    )
    tool_names: Optional[List[str]] = Field(
        default=None,
        description="Specific tool names to register (optional for register operation)"
    )
    recursive: bool = Field(
        default=True,
        description="Whether to scan subdirectories recursively (for discover operation)"
    )
    auto_register_all: bool = Field(
        default=False,
        description="Whether to register all valid discovered tools (for register operation)"
    )
    analysis_tool_name: Optional[str] = Field(
        default=None,
        description="Tool name to analyze (required for analyze operation)"
    )


class ToolManagerWorkflow:
    """Workflow for automatic tool discovery and registration to TCP."""
    
    def __init__(self, model_name: str = "o3"):
        """Initialize the tool manager workflow.
        
        Args:
            model_name: Name of the model to use for tool analysis
        """
        self.model_name = model_name
        self.model = model_manager.get(model_name)
        self.discovered_tools: Dict[str, ToolDiscoveryResult] = {}
        self.registration_results: Dict[str, ToolRegistrationResult] = {}
    
    async def discover_tools_in_directory(self, 
                                        directory_path: str,
                                        recursive: bool = True,
                                        include_patterns: List[str] = None,
                                        exclude_patterns: List[str] = None) -> List[ToolDiscoveryResult]:
        """Discover tools in a directory by scanning Python files.
        
        Args:
            directory_path: Path to directory to scan
            recursive: Whether to scan subdirectories recursively
            include_patterns: File patterns to include (e.g., ['*_tool.py'])
            exclude_patterns: File patterns to exclude (e.g., ['*_test.py'])
            
        Returns:
            List of discovered tools
        """
        logger.info(f"| 🔍 Discovering tools in directory: {directory_path}")
        
        if include_patterns is None:
            include_patterns = ['*_tool.py', '*tool*.py']
        if exclude_patterns is None:
            exclude_patterns = ['*_test.py', '*test*.py', '__pycache__']
        
        discovered_tools = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.warning(f"| ⚠️ Directory {directory_path} does not exist")
            return discovered_tools
        
        # Find Python files
        python_files = []
        if recursive:
            for pattern in include_patterns:
                python_files.extend(directory.rglob(pattern))
        else:
            for pattern in include_patterns:
                python_files.extend(directory.glob(pattern))
        
        # Filter out excluded patterns
        filtered_files = []
        for file_path in python_files:
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                if file_path.match(exclude_pattern):
                    should_exclude = True
                    break
            if not should_exclude:
                filtered_files.append(file_path)
        
        logger.info(f"| 📁 Found {len(filtered_files)} Python files to scan")
        
        # Scan each file for tools
        for file_path in filtered_files:
            try:
                tools_in_file = await self._discover_tools_in_file(str(file_path))
                discovered_tools.extend(tools_in_file)
            except Exception as e:
                logger.error(f"| ❌ Error scanning file {file_path}: {e}")
        
        # Store discovered tools
        for tool_result in discovered_tools:
            self.discovered_tools[tool_result.tool_name] = tool_result
        
        logger.info(f"| ✅ Discovered {len(discovered_tools)} tools")
        return discovered_tools
    
    async def _discover_tools_in_file(self, file_path: str) -> List[ToolDiscoveryResult]:
        """Discover tools in a single Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            List of discovered tools in the file
        """
        discovered_tools = []
        
        try:
            # Convert file path to module path
            relative_path = Path(file_path).relative_to(Path.cwd())
            module_path = str(relative_path).replace('/', '.').replace('\\', '.').replace('.py', '')
            
            # Import the module
            spec = importlib.util.spec_from_file_location(module_path, file_path)
            if spec is None or spec.loader is None:
                logger.warning(f"| ⚠️ Could not load module from {file_path}")
                return discovered_tools
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find tool classes in the module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseTool) and 
                    obj != BaseTool and 
                    hasattr(obj, 'name') and 
                    hasattr(obj, 'description')):
                    
                    # Validate the tool class
                    is_valid, error_message = await self._validate_tool_class(obj)
                    
                    tool_result = ToolDiscoveryResult(
                        tool_name=obj.name,
                        tool_class=obj,
                        module_path=module_path,
                        is_valid=is_valid,
                        error_message=error_message,
                        metadata={
                            'file_path': file_path,
                            'class_name': name,
                            'description': obj.description,
                            'type': getattr(obj, 'type', 'Unknown')
                        }
                    )
                    
                    discovered_tools.append(tool_result)
                    logger.info(f"| 🔧 Found tool: {obj.name} in {file_path}")
        
        except Exception as e:
            logger.error(f"| ❌ Error processing file {file_path}: {e}")
        
        return discovered_tools
    
    async def _validate_tool_class(self, tool_class: Type[BaseTool]) -> tuple[bool, Optional[str]]:
        """Validate a tool class for proper structure.
        
        Args:
            tool_class: Tool class to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check required attributes
            required_attrs = ['name', 'description', 'args_schema']
            for attr in required_attrs:
                if not hasattr(tool_class, attr):
                    return False, f"Missing required attribute: {attr}"
            
            # Check if name is a string
            if not isinstance(tool_class.name, str) or not tool_class.name:
                return False, "Tool name must be a non-empty string"
            
            # Check if description is a string
            if not isinstance(tool_class.description, str) or not tool_class.description:
                return False, "Tool description must be a non-empty string"
            
            # Check if args_schema is a Pydantic model
            if not (hasattr(tool_class.args_schema, '__bases__') and 
                    BaseModel in tool_class.args_schema.__bases__):
                return False, "args_schema must be a Pydantic BaseModel"
            
            # Check if tool has _arun method
            if not hasattr(tool_class, '_arun'):
                return False, "Tool must have _arun method"
            
            return True, None
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    async def register_discovered_tools(self, 
                                      tool_names: Optional[List[str]] = None,
                                      auto_register_all: bool = False) -> List[ToolRegistrationResult]:
        """Register discovered tools to TCP.
        
        Args:
            tool_names: Specific tool names to register (if None, register all valid tools)
            auto_register_all: Whether to register all valid discovered tools
            
        Returns:
            List of registration results
        """
        logger.info(f"| 📝 Registering tools to TCP")
        
        registration_results = []
        
        # Determine which tools to register
        tools_to_register = []
        if tool_names:
            for tool_name in tool_names:
                if tool_name in self.discovered_tools:
                    tools_to_register.append(tool_name)
                else:
                    logger.warning(f"| ⚠️ Tool {tool_name} not found in discovered tools")
        elif auto_register_all:
            tools_to_register = [name for name, result in self.discovered_tools.items() 
                               if result.is_valid]
        else:
            # Register all valid tools
            tools_to_register = [name for name, result in self.discovered_tools.items() 
                               if result.is_valid]
        
        logger.info(f"| 🎯 Registering {len(tools_to_register)} tools")
        
        # Register each tool
        for tool_name in tools_to_register:
            try:
                result = await self._register_single_tool(tool_name)
                registration_results.append(result)
                self.registration_results[tool_name] = result
                
                if result.success:
                    logger.info(f"| ✅ Successfully registered tool: {tool_name}")
                else:
                    logger.error(f"| ❌ Failed to register tool {tool_name}: {result.error_message}")
            
            except Exception as e:
                error_result = ToolRegistrationResult(
                    tool_name=tool_name,
                    success=False,
                    error_message=str(e)
                )
                registration_results.append(error_result)
                self.registration_results[tool_name] = error_result
                logger.error(f"| ❌ Exception registering tool {tool_name}: {e}")
        
        logger.info(f"| ✅ Tool registration completed: {len([r for r in registration_results if r.success])} successful")
        return registration_results
    
    async def _register_single_tool(self, tool_name: str) -> ToolRegistrationResult:
        """Register a single tool to TCP.
        
        Args:
            tool_name: Name of the tool to register
            
        Returns:
            Registration result
        """
        if tool_name not in self.discovered_tools:
            return ToolRegistrationResult(
                tool_name=tool_name,
                success=False,
                error_message="Tool not found in discovered tools"
            )
        
        tool_result = self.discovered_tools[tool_name]
        
        if not tool_result.is_valid:
            return ToolRegistrationResult(
                tool_name=tool_name,
                success=False,
                error_message=f"Tool validation failed: {tool_result.error_message}"
            )
        
        try:
            # Check if tool is already registered
            existing_tool = tcp.get_info(tool_name)
            if existing_tool:
                return ToolRegistrationResult(
                    tool_name=tool_name,
                    success=True,
                    was_already_registered=True,
                    error_message="Tool was already registered"
                )
            
            # Create tool instance
            tool_instance = tool_result.tool_class()
            
            # Create ToolInfo
            tool_info = ToolInfo(
                name=tool_instance.name,
                type=tool_instance.type,
                description=tool_instance.description,
                args_schema=tool_instance.args_schema,
                metadata=tool_instance.metadata,
                cls=tool_result.tool_class,
                instance=tool_instance
            )
            
            # Register with TCP
            await tcp.register(tool_info)
            
            return ToolRegistrationResult(
                tool_name=tool_name,
                success=True,
                was_already_registered=False
            )
        
        except Exception as e:
            return ToolRegistrationResult(
                tool_name=tool_name,
                success=False,
                error_message=str(e)
            )
    
    async def analyze_tool_usage(self, tool_name: str) -> Dict[str, Any]:
        """Analyze tool usage patterns and provide recommendations.
        
        Args:
            tool_name: Name of the tool to analyze
            
        Returns:
            Analysis results
        """
        if tool_name not in self.discovered_tools:
            return {"error": f"Tool {tool_name} not found in discovered tools"}
        
        tool_result = self.discovered_tools[tool_name]
        
        # Use LLM to analyze the tool
        analysis_prompt = f"""Analyze the following tool and provide insights:

Tool Name: {tool_result.tool_name}
Description: {tool_result.metadata.get('description', 'N/A')}
Type: {tool_result.metadata.get('type', 'N/A')}

Please provide:
1. Tool complexity assessment (Simple/Medium/Complex)
2. Potential use cases
3. Integration recommendations
4. Performance considerations
5. Security considerations

Format as JSON with these fields: complexity, use_cases, integration_tips, performance_notes, security_notes"""

        try:
            from langchain_core.messages import HumanMessage
            response = await self.model.ainvoke([HumanMessage(content=analysis_prompt)])
            
            # Try to parse JSON response
            import json
            analysis = json.loads(response.content)
            
            return {
                "tool_name": tool_name,
                "analysis": analysis,
                "metadata": tool_result.metadata
            }
        
        except Exception as e:
            logger.error(f"| ❌ Error analyzing tool {tool_name}: {e}")
            return {
                "tool_name": tool_name,
                "error": str(e),
                "metadata": tool_result.metadata
            }
    
    def get_discovery_summary(self) -> Dict[str, Any]:
        """Get summary of tool discovery results.
        
        Returns:
            Discovery summary
        """
        total_tools = len(self.discovered_tools)
        valid_tools = len([r for r in self.discovered_tools.values() if r.is_valid])
        invalid_tools = total_tools - valid_tools
        
        registered_tools = len([r for r in self.registration_results.values() if r.success])
        
        return {
            "total_discovered": total_tools,
            "valid_tools": valid_tools,
            "invalid_tools": invalid_tools,
            "registered_tools": registered_tools,
            "discovery_results": {
                name: {
                    "is_valid": result.is_valid,
                    "error_message": result.error_message,
                    "module_path": result.module_path
                }
                for name, result in self.discovered_tools.items()
            },
            "registration_results": {
                name: {
                    "success": result.success,
                    "error_message": result.error_message,
                    "was_already_registered": result.was_already_registered
                }
                for name, result in self.registration_results.items()
            }
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        self.discovered_tools.clear()
        self.registration_results.clear()
        logger.info("| 🧹 Tool manager workflow cleaned up")


@tcp.tool()
class ToolManagerTCPTool(BaseTool):
    """TCP tool for automatic tool discovery and registration."""
    
    name: str = "tool_manager"
    type: str = "Tool Manager"
    description: str = """Tool Manager for automatic discovery and registration of tools to TCP.
    
    This tool provides comprehensive tool management capabilities:
    
    1. **Discover Tools**: Scan directories for tool classes and validate them
       - Scans Python files for BaseTool subclasses
       - Validates tool structure and requirements
       - Supports recursive directory scanning
       - Filters files by patterns (include/exclude)
    
    2. **Register Tools**: Register discovered tools to TCP
       - Registers valid tools to the TCP server
       - Handles duplicate registration gracefully
       - Provides detailed registration results
    
    3. **Analyze Tools**: Get detailed analysis of tool usage and recommendations
       - Uses LLM to analyze tool complexity and use cases
       - Provides integration recommendations
       - Identifies performance and security considerations
    
    4. **Summary**: Get comprehensive summary of discovery and registration results
       - Shows statistics of discovered vs registered tools
       - Lists validation errors and registration failures
       - Provides overview of tool management operations
    
    5. **Cleanup**: Clean up tool manager state
       - Clears discovered tools cache
       - Resets registration results
       - Frees up memory resources
    
    Operations:
    - discover: Scan directory for tools
    - register: Register discovered tools to TCP
    - analyze: Analyze specific tool
    - summary: Get operation summary
    - cleanup: Clean up manager state
    """
    args_schema: Type[ToolManagerArgs] = ToolManagerArgs
    metadata: Dict[str, Any] = {}
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        """Initialize the tool manager TCP tool."""
        model_name = model_name or config.tool_manager_tool.get("model_name", "o3")
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name
        
        # Initialize the workflow
        self.workflow = ToolManagerWorkflow(model_name=model_name)
    
    async def _arun(self, 
                   operation: str,
                   directory_path: Optional[str] = None,
                   tool_names: Optional[List[str]] = None,
                   recursive: bool = True,
                   auto_register_all: bool = False,
                   analysis_tool_name: Optional[str] = None) -> ToolResponse:
        """Execute tool manager operations.
        
        Args:
            operation: Operation to perform
            directory_path: Directory to scan for tools
            tool_names: Specific tool names to register
            recursive: Whether to scan recursively
            auto_register_all: Whether to register all valid tools
            analysis_tool_name: Tool name to analyze
            
        Returns:
            Tool execution result
        """
        try:
            logger.info(f"| 🛠️ Tool Manager operation: {operation}")
            
            if operation == "discover":
                if not directory_path:
                    return ToolResponse(content="Error: directory_path is required for discover operation")
                
                # Discover tools in directory
                discovered_tools = await self.workflow.discover_tools_in_directory(
                    directory_path=directory_path,
                    recursive=recursive
                )
                
                result = {
                    "operation": "discover",
                    "success": True,
                    "directory_path": directory_path,
                    "recursive": recursive,
                    "discovered_count": len(discovered_tools),
                    "tools": [
                        {
                            "name": tool.tool_name,
                            "is_valid": tool.is_valid,
                            "error_message": tool.error_message,
                            "module_path": tool.module_path,
                            "metadata": tool.metadata
                        }
                        for tool in discovered_tools
                    ]
                }
                
                return ToolResponse(content=str(result))
            
            elif operation == "register":
                # Register discovered tools
                registration_results = await self.workflow.register_discovered_tools(
                    tool_names=tool_names,
                    auto_register_all=auto_register_all
                )
                
                result = {
                    "operation": "register",
                    "success": True,
                    "registered_count": len([r for r in registration_results if r.success]),
                    "total_attempted": len(registration_results),
                    "results": [
                        {
                            "tool_name": result.tool_name,
                            "success": result.success,
                            "error_message": result.error_message,
                            "was_already_registered": result.was_already_registered
                        }
                        for result in registration_results
                    ]
                }
                
                return ToolResponse(content=str(result))
            
            elif operation == "analyze":
                if not analysis_tool_name:
                    return ToolResponse(content="Error: analysis_tool_name is required for analyze operation")
                
                # Analyze specific tool
                analysis_result = await self.workflow.analyze_tool_usage(analysis_tool_name)
                
                result = {
                    "operation": "analyze",
                    "success": True,
                    "tool_name": analysis_tool_name,
                    "analysis": analysis_result
                }
                
                return ToolResponse(content=str(result))
            
            elif operation == "summary":
                # Get summary of operations
                summary = self.workflow.get_discovery_summary()
                
                result = {
                    "operation": "summary",
                    "success": True,
                    "summary": summary
                }
                
                return ToolResponse(content=str(result))
            
            elif operation == "cleanup":
                # Cleanup workflow state
                await self.workflow.cleanup()
                
                result = {
                    "operation": "cleanup",
                    "success": True,
                    "message": "Tool manager workflow cleaned up"
                }
                
                return ToolResponse(content=str(result))
            
            else:
                return ToolResponse(content=f"Error: Unknown operation '{operation}'. Available operations: discover, register, analyze, summary, cleanup")
        
        except Exception as e:
            logger.error(f"| ❌ Tool Manager operation failed: {e}")
            return ToolResponse(content=f"Error in tool manager operation: {str(e)}")
    
    def _run(self, 
             operation: str,
             directory_path: Optional[str] = None,
             tool_names: Optional[List[str]] = None,
             recursive: bool = True,
             auto_register_all: bool = False,
             analysis_tool_name: Optional[str] = None) -> ToolResponse:
        """Execute tool manager operations synchronously (fallback)."""
        try:
            # Run async version
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(
                    operation=operation,
                    directory_path=directory_path,
                    tool_names=tool_names,
                    recursive=recursive,
                    auto_register_all=auto_register_all,
                    analysis_tool_name=analysis_tool_name
                ))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous execution: {e}")
            return ToolResponse(content=f"Error in synchronous execution: {e}")


# Global tool manager workflow instance
tool_manager_workflow = ToolManagerWorkflow()