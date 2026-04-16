"""Environment to Tool (E2T) Transformer.

Converts ECP environments to TCP tools.
"""
from typing import Any, Dict
from pydantic import ConfigDict
import inspect

from src.logger import logger
from src.tool.server import tool_manager
from src.environment.server import environment_manager
from src.transformation.types import E2TRequest, E2TResponse
from src.tool.types import Tool, ToolResponse


def create_wrapped_tool_class(action_config, env_config, env_name):
    # Capture variables in closure to avoid scope issues
    tool_name = f"{env_name}.{action_config.name}"
    tool_description = action_config.description
    
    class WrappedTool(Tool):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
        
        name: str = tool_name
        description: str = tool_description
        metadata: Dict[str, Any] = action_config.metadata
        version: str = action_config.version
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        async def __call__(self, **kwargs) -> ToolResponse:
            """Execute the wrapped action."""
            try:
                action_function = action_config.function if action_config else None
                if action_function is None:
                    return ToolResponse(
                        success=False,
                        message=f"Action {action_config.name} has no function"
                    )
                
                # Check if function is bound or unbound
                if hasattr(action_function, '__self__'):
                    # Bound method: call directly without passing instance
                    result = await action_function(**kwargs)
                else:
                    # Unbound method: get instance and pass as first argument
                    env_instance = env_config.instance
                    if env_instance is None:
                        env_instance = await environment_manager.get(env_config.name)
                        if env_instance is None:
                            return ToolResponse(
                                success=False,
                                message=f"Environment {env_config.name} instance not found"
                            )
                    result = await action_function(env_instance, **kwargs)
                
                # Convert result to ToolResponse if needed
                if isinstance(result, ToolResponse):
                    return result
                elif isinstance(result, dict):
                    return ToolResponse(
                        success=result.get("success", True),
                        message=result.get("message", str(result)),
                        extra=result.get("extra")
                    )
                else:
                    return ToolResponse(
                        success=True,
                        message=str(result)
                    )
            except Exception as e:
                return ToolResponse(
                    success=False,
                    message=f"Error executing action: {str(e)}"
                )
    
    # Set docstring and signature from action function
    WrappedTool.__call__.__doc__ = inspect.getdoc(action_config.function)
    WrappedTool.__call__.__signature__ = inspect.signature(action_config.function)
    
    return WrappedTool


class E2TTransformer:
    """Transformer for converting ECP environments to TCP tools."""
    
    async def transform(self, request: E2TRequest) -> E2TResponse:
        """Convert ECP environments to TCP tools.
        
        Args:
            request (E2TRequest): E2TRequest instance
            
        Returns:
            E2TResponse: E2TResponse
        """
        try:
            logger.info("| 🔧 ECP to TCP transformation")
            for env_name in request.env_names:
                env_config = await environment_manager.get_info(env_name)
                
                actions = env_config.actions
                for action_name, action_config in actions.items():
                    
                    WrappedToolClass = create_wrapped_tool_class(action_config, env_config, env_name)
                    await tool_manager.register(WrappedToolClass, config={}, override=True)
                    logger.info(f"| ✅ E2T: Tool {env_name}.{action_name} added to TCP")
                        
            return E2TResponse(
                success=True,
                message="ECP to TCP transformation completed",
            )
            
        except Exception as e:
            logger.error(f"| ❌ ECP to TCP transformation failed: {e}")
            return E2TResponse(
                success=False,
                message="ECP to TCP transformation failed: " + str(e)
            )
