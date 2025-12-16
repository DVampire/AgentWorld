"""Tool to Agent (T2A) Transformer.

Converts TCP tools to ACP agents.
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field, ConfigDict

from src.config import config
from src.logger import logger
from src.tool.server import tcp
from src.agent.server import acp
from src.agent import ToolCallingAgent
from src.model import model_manager
from src.utils import dedent
from src.transformation.types import T2ARequest, T2AResponse


def create_composed_agent_class(selected_tool_infos, agent_name: str, agent_description: str, metadata_: Dict[str, Any]):
    """Create a composed agent class from multiple tools.
    
    Args:
        selected_tool_infos: List of tool configurations
        agent_name: Name of the composed agent
        agent_description: Description of the composed agent
        metadata_: Metadata for the agent
        
    Returns:
        Type[ToolCallingAgent]: The composed agent class
    """
    class ComposedAgentInputArgs(BaseModel):
        task: str = Field(description="The task to complete.")
        files: Optional[List[str]] = Field(default=None, description="The files to attach to the task.")
    
    class ComposedAgent(ToolCallingAgent):
        name: str = Field(default=agent_name, description="The name of the composed agent")
        type: str = Field(default="Composed Agent", description="The type of the composed agent")
        description: str = Field(default=agent_description, description="The description of the composed agent")
        args_schema: Type[ComposedAgentInputArgs] = Field(default=ComposedAgentInputArgs, description="The args schema of the composed agent.")
        metadata: Dict[str, Any] = Field(default=metadata_, description="The metadata of the composed agent")
        
        model_config = ConfigDict(
            arbitrary_types_allowed=True, 
            extra="allow"
        )
        
        def __init__(self, 
                    workdir: str,
                    model_name: Optional[str] = None,
                    prompt_name: Optional[str] = None,
                    max_steps: int = 20,
                    review_steps: int = 5,
                    log_max_length: int = 1000,
                    **kwargs):
            
            # Set default prompt name for tool calling
            if not prompt_name:
                prompt_name = "tool_calling"
            
            super().__init__(
                workdir=workdir,
                model_name=model_name,
                prompt_name=prompt_name,
                max_steps=max_steps,
                review_steps=review_steps,
                log_max_length=log_max_length,
                **kwargs)
    
    return ComposedAgent


class T2ATransformer:
    """Transformer for converting TCP tools to ACP agents."""
    
    async def transform(self, request: T2ARequest) -> T2AResponse:
        """Convert TCP tools to ACP agents.
        
        This function takes multiple TCP tools and creates an agent that can use them.
        
        Args:
            request (T2ARequest): T2ARequest instance with tool names
            
        Returns:
            T2AResponse: T2AResponse with success status and message
        """
        try:
            logger.info("| 🔧 TCP to ACP transformation")
            
            # Step 1: Collect selected tool information
            selected_tool_infos = []
            for tool_name in request.tool_names:
                tool_info = await tcp.get_info(tool_name)
                if tool_info:
                    selected_tool_infos.append(tool_info)
                else:
                    logger.warning(f"| ⚠️ Tool {tool_name} not found in TCP")
            
            if not selected_tool_infos:
                return T2AResponse(
                    success=False,
                    message="No valid tools found for transformation"
                )
            
            # Step 2: Use LLM to generate agent information
            class DynamicComposedArgs(BaseModel):
                name: str = Field(description="The name of the composed agent, the name should be a snake_case string.")
                description: str = Field(description="The description of the composed agent, the description should be a concise description of the agent.")
            
            tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in selected_tool_infos])
            prompt = dedent(f"""
                You are a helpful assistant that composes an agent from a list of tools.
                
                The tools are:
                {tool_descriptions}
                
                Please compose an agent and give the name and description of the agent.
            """)
            
            from src.message.types import HumanMessage
            response = await model_manager(
                model="gpt-4.1",
                messages=[HumanMessage(content=prompt)],
                response_format=DynamicComposedArgs
            )
            response = response.extra["parsed_model"]
            
            agent_name = response.name
            agent_description = response.description
            metadata_ = {}
            
            # Step 3: Create agent class
            ComposedAgent = create_composed_agent_class(
                selected_tool_infos,
                agent_name,
                agent_description,
                metadata_
            )
            
            # Step 4: Register the agent class
            await acp.register(ComposedAgent, config={"workdir": config.workdir}, override=True)
            
            logger.info(f"| ✅ T2A: Agent {agent_name} created with {len(selected_tool_infos)} tools")
            
            return T2AResponse(
                success=True,
                message=f"Successfully created agent {agent_name} with {len(selected_tool_infos)} tools"
            )
            
        except Exception as e:
            logger.error(f"| ❌ TCP to ACP transformation failed: {e}")
            return T2AResponse(
                success=False,
                message="TCP to ACP transformation failed: " + str(e)
            )
