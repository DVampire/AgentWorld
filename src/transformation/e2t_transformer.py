"""Environment to Tool (E2T) Transformer.

Converts ECP environments to TCP tools.
"""
from typing import Any, Dict, Type
from pydantic import BaseModel

from src.logger import logger
from src.tool.server import tcp
from src.environment.server import ecp
from src.transformation.types import E2TRequest, E2TResponse
from src.tool.types import Tool, ToolResponse


def create_wrapped_tool_class(action_config, env_config, env_name):
    # Capture variables in closure to avoid scope issues
    tool_name = f"{env_name}.{action_config.name}"
    tool_description = action_config.description
    
    class WrappedTool(Tool):
        name: str = tool_name
        description: str = tool_description
        enabled: bool = True
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        @property
        def args_schema(self) -> Type[BaseModel]:
            """Return the BaseModel type from action config's args_schema."""
            if action_config:
                try:
                    # Access the property which will compute if needed
                    schema = action_config.args_schema
                    # Ensure it's a valid BaseModel subclass
                    if schema is not None and isinstance(schema, type) and issubclass(schema, BaseModel):
                        return schema
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to get args_schema from action_config for {tool_name}: {e}")
                    import traceback
                    logger.debug(f"| Traceback: {traceback.format_exc()}")
            # Fallback to building from parameter_schema (which will parse __call__ signature)
            # But WrappedTool.__call__ has **kwargs, so it will return empty schema
            # So we should use action_config's schema instead
            try:
                if action_config:
                    # Try to compute from action_config's parameter_schema directly
                    from src.utils.parameter_utils import build_args_schema
                    schema = action_config.parameter_schema
                    computed_schema = build_args_schema(action_config.name, schema)
                    if computed_schema is not None:
                        return computed_schema
            except Exception as e:
                logger.debug(f"| ⚠️ Failed to compute args_schema from action_config parameter_schema: {e}")
            # Final fallback
            return super().args_schema
        
        @property
        def function_calling(self) -> Dict[str, Any]:
            """Return the function calling representation from action config."""
            if action_config:
                try:
                    # Access the property which will compute if needed
                    fc = action_config.function_calling
                    # Ensure it's a valid dict
                    if fc is not None and isinstance(fc, dict):
                        return fc
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to get function_calling from action_config for {tool_name}: {e}")
                    import traceback
                    logger.debug(f"| Traceback: {traceback.format_exc()}")
            # Fallback to building from parameter_schema
            return super().function_calling
        
        @property
        def text(self) -> str:
            """Return the text representation from action config."""
            if action_config:
                try:
                    # Access the property which will compute if needed
                    txt = action_config.text
                    # Ensure it's a valid string
                    if txt is not None and isinstance(txt, str):
                        return txt
                except Exception as e:
                    logger.warning(f"| ⚠️ Failed to get text from action_config for {tool_name}: {e}")
                    import traceback
                    logger.debug(f"| Traceback: {traceback.format_exc()}")
            # Fallback to building from parameter_schema
            return super().text
        
        async def __call__(self, **kwargs) -> ToolResponse:
            """Execute the wrapped action.
            
            Args are passed as keyword arguments, matching the action function's signature.
            The tool_context_manager calls tool(**input), which unpacks the input dict as kwargs.
            """
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
                        env_instance = await ecp.get(env_config.name)
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
                    
                    WrappedToolClass = create_wrapped_tool_class(action_config, env_config, env_name)
                    
                    # Don't pass action_config and env_config in config - they contain non-serializable functions
                    # WrappedTool.__init__ will get them from closure if not in config
                    # The args_schema, function_calling, and text will be correctly retrieved from action_config
                    # when the tool instance is created, and saved in ToolConfig
                    await tcp.register(WrappedToolClass, config={}, override=True)
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
