"""Environment to Tool (E2T) Transformer.

Converts ECP environments to TCP tools.
"""

import asyncio
from typing import Any, Dict, Type
from pydantic import BaseModel

from src.logger import logger
from src.tool.server import tcp
from src.tool.types import Tool, ToolResponse
from src.environment.server import ecp
from src.transformation.types import E2TRequest, E2TResponse


def create_wrapped_tool_class(action_config, env_config):
    # Capture variables in closure to avoid scope issues
    tool_name = action_config.name
    tool_description = action_config.description
    
    class WrappedTool(Tool):
        name: str = tool_name
        description: str = tool_description
        enabled: bool = True
        
        def __init__(self, **kwargs):
            # Ensure name is set if not provided
            if 'name' not in kwargs:
                kwargs['name'] = tool_name
            super().__init__(**kwargs)
            # Get action config and env config from config if available
            # This allows the tool to work when instantiated from the registered class
            init_config = kwargs.get('config', {}) or {}
            self._action_config = init_config.get('action_config', action_config)
            self._env_config = init_config.get('env_config', env_config)
        
        @property
        def args_schema(self) -> Type[BaseModel]:
            """Return the BaseModel type from action config's args_schema."""
            if self._action_config:
                schema = self._action_config.args_schema
                # Ensure it's a valid BaseModel subclass
                if schema is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
                    return schema
                # If invalid, fall back to building from parameter_schema
                return super().args_schema
            # Fallback to default if no action config
            return super().args_schema
        
        async def __call__(self, input: Dict[str, Any], **kwargs) -> ToolResponse:
            """Execute the wrapped action.
            
            Args are passed as keyword arguments, matching the action function's signature.
            The tool_context_manager calls tool(**input), which unpacks the input dict as kwargs.
            """
            try:
                action_function = self._action_config.function
                if action_function is None:
                    return ToolResponse(
                        success=False,
                        message=f"Action {self._action_config.name} has no function"
                    )
                
                # Get environment instance if needed
                env_instance = self._env_config.instance
                if env_instance is None:
                    # Try to get instance from ECP
                    env_instance = await ecp.get(self._env_config.name)
                
                # Check if function is bound or unbound
                if hasattr(action_function, '__self__'):
                    # Bound method: call directly
                    if asyncio.iscoroutinefunction(action_function):
                        result = await action_function(**kwargs)
                    else:
                        result = action_function(**kwargs)
                else:
                    # Unbound method: pass instance as first argument if needed
                    if env_instance is not None:
                        if asyncio.iscoroutinefunction(action_function):
                            result = await action_function(env_instance, **kwargs)
                        else:
                            result = action_function(env_instance, **kwargs)
                    else:
                        if asyncio.iscoroutinefunction(action_function):
                            result = await action_function(**kwargs)
                        else:
                            result = action_function(**kwargs)
                
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
                env_config = await ecp.get_info(env_name)
                
                actions = env_config.actions
                for action_name, action_config in actions.items():
                    
                    WrappedToolClass = create_wrapped_tool_class(action_config, env_config)
                    
                    await tcp.register(WrappedToolClass, config={}, override=True)
                    logger.info(f"| ✅ E2T: Tool {action_name} added to TCP")
                        
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
