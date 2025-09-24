"""Transformation server for protocol conversions.

This server handles transformations between ECP, TCP, and ACP protocols.
"""

import asyncio
from typing import Any, List, Optional, Callable
from langchain.tools import StructuredTool

from src.logger import logger

from src.environments.protocol.types import ActionInfo
from src.environments.protocol.server import ecp
from src.agents.protocol.server import acp
from src.tools.protocol.server import tcp
from src.tools.protocol.tool import WrappedTool
from src.tools.protocol.types import ToolInfo

from src.transformation.protocol.types import (
    TransformationType,
    E2TRequest,
    E2TResponse,
)


class TransformationServer:
    """Server for handling protocol transformations between ECP, TCP, and ACP."""
    
    def __init__(self):
        """Initialize the transformation server.
        
        Args:
            config: Configuration for transformations
        """
        logger.info("| 🔄 Transformation Server initialized")
    
    async def transform(self, 
                        type: str,
                        env_names: Optional[List[str]] = None,
                        tool_names: Optional[List[str]] = None,
                        agent_names: Optional[List[str]] = None,
                        ) -> Any:
        """Perform a protocol transformation.
        
        Args:
            request: Transformation request
            
        Returns:
            Transformation response
        """
        try:
            logger.info(f"| 🔄 Starting transformation: {type}")
            
            # Route to appropriate transformation method

            if type == TransformationType.E2T.value:
                request = E2TRequest(
                    type=type,
                    env_names=env_names
                )
                result = await self._e2t(request)
            else:
                raise ValueError(f"Unknown transformation type: {type}")
            
            logger.info(f"| ✅ Transformation completed: {type}")
            return result
            
        except Exception as e:
            logger.error(f"| ❌ Transformation failed: {e}")
            return "Transformation failed: " + str(e)
    
    async def _e2t(self, request: E2TRequest) -> E2TResponse:
        """Convert ECP environments to TCP tools.
        
        Args:
            request (E2TRequest): E2TRequest instance
            
        Returns:
            E2TResponse: E2TResponse
        """
        def make_wrapped_func(env_info, action_info):
            if asyncio.iscoroutinefunction(action_info.function):
                async def _async_action_wrapper(**kwargs):
                    return await action_info.function(env_info.instance, **kwargs)
                return _async_action_wrapper
            else:
                def _sync_action_wrapper(**kwargs):
                    return action_info.function(env_info.instance, **kwargs)
                return _sync_action_wrapper
        
        try:
            logger.info("| 🔧 ECP to TCP transformation")
            for env_name in request.env_names:
                env_info = ecp.get_info(env_name)
                
                actions = env_info.actions
                for action_name, action_info in actions.items():
                    # Create Tool
                    tool = StructuredTool(
                        name=action_name,
                        description=action_info.description,
                        args_schema=action_info.args_schema,
                        func=make_wrapped_func(env_info, action_info),
                        coroutine=make_wrapped_func(env_info, action_info) if asyncio.iscoroutinefunction(action_info.function) else None,
                        metadata=action_info.metadata
                    )
                    tool = WrappedTool(tool=tool)
                    
                    # Create ToolInfo
                    tool_info = ToolInfo(
                        name=action_name,
                        type=action_info.type,
                        description=action_info.description,
                        args_schema=action_info.args_schema,
                        metadata=action_info.metadata,
                        cls=WrappedTool,
                        instance=None
                    )
                    tool_info.instance = tool
                    
                    await tcp.add(tool_info)
                    logger.info(f"| ✅ E2T: Tool {tool.name} added to TCP")
                        
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
            
transformation = TransformationServer()