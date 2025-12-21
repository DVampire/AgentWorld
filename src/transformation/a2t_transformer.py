"""Agent to Tool (A2T) Transformer.

Converts ACP agents to TCP tools.
"""

import asyncio
from typing import Any, Dict
from typing import Optional, List
from pydantic import ConfigDict

from src.logger import logger
from src.tool.server import tcp
from src.tool.types import Tool, ToolResponse
from src.agent.server import acp
from src.transformation.types import A2TRequest, A2TResponse


def create_wrapped_tool_class(agent_config, agent_instance):
    class WrappedTool(Tool):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
        
        name: str = agent_config.name
        description: str = agent_config.description
        metadata: Dict[str, Any] = agent_config.metadata
        version: str = agent_config.version
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Get agent info and instance from config if available
            # This allows the tool to work when instantiated from the registered class
            config = kwargs.get('config', {}) or {}
            self._agent_config = config.get('agent_config', agent_config)
            self._agent_instance = config.get('agent_instance', agent_instance)
        
        async def __call__(self, task: str, files: Optional[List[str]] = None, **kwargs) -> ToolResponse:
            """Wrapper function to call agent.
            
            Args:
                task (str): The task to complete
                files (Optional[List[str]]): The files to attach to the task
            """
            try:
                # Extract task and files from input
                # Call agent directly using __call__ method
                result = await self._agent_instance(task=task, files=files, **kwargs)
                
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
                    message=f"Error executing agent: {str(e)}"
                )
    
    return WrappedTool

class A2TTransformer:
    """Transformer for converting ACP agents to TCP tools."""
    
    async def transform(self, request: A2TRequest) -> A2TResponse:
        """Convert ACP agents to TCP tools.
        
        Args:
            request (A2TRequest): A2TRequest instance
            
        Returns:
            A2TResponse: A2TResponse
        """
        
        try:
            logger.info("| 🔧 ACP to TCP transformation")
            
            selected_agent_configs = []
            for agent_name in request.agent_names:
                agent_config = await acp.get_info(agent_name)
                
                if agent_config:
                    selected_agent_configs.append(agent_config)
                else:
                    logger.warning(f"| ⚠️ Agent {agent_name} not found in ACP")
                    
            if not selected_agent_configs:
                return A2TResponse(
                    success=False,
                    message="No valid agents found for transformation"
                )
                
            for agent_config in selected_agent_configs:
                # Get agent instance
                agent_instance = agent_config.instance
                if agent_instance is None:
                    logger.warning(f"| ⚠️ Agent {agent_config.name} has no instance")
                    continue
                
                WrappedToolClass = create_wrapped_tool_class(agent_config, agent_instance)

                await tcp.register(WrappedToolClass, config={}, override=True)
                logger.info(f"| ✅ ACP to TCP transformation completed: {agent_config.name}")
                
            logger.info(f"| ✅ ACP to TCP transformation completed: {len(selected_agent_configs)} tools")
            
            return A2TResponse(
                success=True,
                message=f"Successfully converted {len(selected_agent_configs)} agents to tools"
            )
            
        except Exception as e:
            logger.error(f"| ❌ ACP to TCP transformation failed: {e}")
            return A2TResponse(
                success=False,
                message="ACP to TCP transformation failed: " + str(e)
            )
