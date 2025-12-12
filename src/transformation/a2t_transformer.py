"""Agent to Tool (A2T) Transformer.

Converts ACP agents to TCP tools.
"""

import asyncio
from typing import Any, Dict

from src.logger import logger
from src.tool.server import tcp
from src.tool.types import Tool, ToolResponse, ToolConfig
from src.agent.server import acp
from src.transformation.types import A2TRequest, A2TResponse


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
            
            selected_agent_infos = []
            for agent_name in request.agent_names:
                agent_info = await acp.get_info(agent_name)
                
                if agent_info:
                    selected_agent_infos.append(agent_info)
                else:
                    logger.warning(f"| ⚠️ Agent {agent_name} not found in ACP")
                    
            if not selected_agent_infos:
                return A2TResponse(
                    success=False,
                    message="No valid agents found for transformation"
                )
                
            for agent_info in selected_agent_infos:
                # Create a dynamic Tool class for this agent
                # Use default arguments to capture loop variables correctly
                def create_wrapped_tool_class(info=agent_info):
                    class WrappedTool(Tool):
                        name: str = info.name
                        description: str = info.description
                        enabled: bool = True
                        
                        def __init__(self, **kwargs):
                            super().__init__(**kwargs)
                            self._agent_info = info
                        
                        async def __call__(self, input: Dict[str, Any]) -> ToolResponse:
                            """Execute the wrapped agent."""
                            try:
                                # Extract task and files from input
                                task = input.get("task", "")
                                files = input.get("files", None)
                                
                                if asyncio.iscoroutinefunction(self._agent_info.instance.ainvoke):
                                    result = await self._agent_info.instance.ainvoke(task=task, files=files)
                                else:
                                    result = self._agent_info.instance.invoke(task=task, files=files)
                                
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
                
                WrappedToolClass = create_wrapped_tool_class()
                wrapped_tool = WrappedToolClass()
                
                tool_info = ToolConfig(
                    id=0,  # Will be assigned by TCP
                    name=agent_info.name,
                    description=agent_info.description,
                    enabled=True,
                    version="1.0.0",
                    cls=WrappedToolClass,
                    config={},
                    instance=wrapped_tool,
                    metadata=agent_info.metadata or {},
                    args_schema=agent_info.args_schema
                )
                
                await tcp.register(tool_info)
                logger.info(f"| ✅ ACP to TCP transformation completed: {agent_info.name}")
                
            logger.info(f"| ✅ ACP to TCP transformation completed: {len(selected_agent_infos)} tools")
            
            return A2TResponse(
                success=True,
                message=f"Successfully converted {len(selected_agent_infos)} agents to tools"
            )
            
        except Exception as e:
            logger.error(f"| ❌ ACP to TCP transformation failed: {e}")
            return A2TResponse(
                success=False,
                message="ACP to TCP transformation failed: " + str(e)
            )
