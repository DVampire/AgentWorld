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
                env_config = ecp.get_info(env_name)
                
                actions = env_config.actions
                for action_name, action_config in actions.items():
                    # Create a dynamic Tool class for this action
                    # Use default arguments to capture loop variables correctly
                    def create_wrapped_tool_class(action_name_val=action_name, config=action_config, env=env_config):
                        # Capture variables in closure to avoid scope issues
                        tool_name = action_name_val
                        tool_description = config.description
                        action_cfg = config
                        env_cfg = env
                        
                        class WrappedTool(Tool):
                            name: str = tool_name
                            description: str = tool_description
                            enabled: bool = True
                            
                            def __init__(self, **kwargs):
                                # Ensure name is set if not provided
                                if 'name' not in kwargs:
                                    kwargs['name'] = tool_name
                                super().__init__(**kwargs)
                                self._action_config = action_cfg
                                self._env_config = env_cfg
                            
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
                            
                            async def __call__(self, **kwargs) -> ToolResponse:
                                """Execute the wrapped action.
                                
                                Args are passed as keyword arguments, matching the action function's signature.
                                The tool_context_manager calls tool(**input), which unpacks the input dict as kwargs.
                                """
                                try:
                                    action_function = self._action_config.function
                                    if hasattr(action_function, '__self__'):
                                        # Bound method: call directly without passing instance
                                        result = await action_function(**kwargs)
                                    else:
                                        # Unbound method: pass instance as first argument
                                        result = await action_function(self._env_config.instance, **kwargs)
                                    
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
                    
                    WrappedToolClass = create_wrapped_tool_class()
                    wrapped_tool = WrappedToolClass()
                    
                    # Register the tool instance directly (tcp.register expects Tool instance or class, not ToolConfig)
                    # The ToolConfig will be created internally by tcp.register
                    await tcp.register(wrapped_tool, override=True)
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
