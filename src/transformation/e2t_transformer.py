"""Environment to Tool (E2T) Transformer.

Converts ECP environments to TCP tools.
"""

import asyncio
from typing import Any, Dict

from src.logger import logger
from src.tool.server import tcp
from src.tool.types import Tool, ToolResponse, ToolConfig
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
                            
                            async def __call__(self, input: Dict[str, Any]) -> ToolResponse:
                                """Execute the wrapped action."""
                                try:
                                    if asyncio.iscoroutinefunction(self._action_config.function):
                                        result = await self._action_config.function(self._env_config.instance, **input)
                                    else:
                                        result = self._action_config.function(self._env_config.instance, **input)
                                    
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
                    
                    # Get args_schema from metadata if available
                    args_schema = None
                    if action_config.metadata and 'args_schema' in action_config.metadata:
                        args_schema = action_config.metadata['args_schema']
                    elif hasattr(action_config, 'args_schema') and action_config.args_schema:
                        args_schema = action_config.args_schema
                    
                    # Create ToolConfig
                    tool_info = ToolConfig(
                        id=0,  # Will be assigned by TCP
                        name=action_name,
                        description=action_config.description,
                        enabled=True,
                        version="1.0.0",
                        cls=WrappedToolClass,
                        config={},
                        instance=wrapped_tool,
                        metadata=action_config.metadata or {},
                        args_schema=args_schema
                    )
                    
                    await tcp.register(tool_info)
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
