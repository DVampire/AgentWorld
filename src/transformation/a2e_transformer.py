"""Agent to Environment (A2E) Transformer.

Converts ACP agents to ECP environments.
"""

from typing import Any, Dict, Type, Optional, List
from pydantic import BaseModel, Field, ConfigDict

from src.logger import logger
from src.environment.types import Environment, EnvironmentConfig, ActionConfig, ActionResult
from src.environment.server import ecp
from src.agent.server import acp
from src.agent.types import AgentResponse
from src.model import model_manager
from src.utils import dedent
from src.transformation.types import A2ERequest, A2EResponse


def create_composed_environment_class(selected_agent_infos, env_name: str, env_description: str, metadata_: Dict[str, Any]):
    """Create a composed environment class from multiple agents.
    
    Args:
        selected_agent_infos: List of agent configurations
        env_name: Name of the composed environment
        env_description: Description of the composed environment
        metadata_: Metadata for the environment
        
    Returns:
        Type[Environment]: The composed environment class
    """
    # Create actions for each agent
    actions = {}
    for agent_info in selected_agent_infos:
        # Get agent instance
        agent_instance = agent_info.instance
        if agent_instance is None:
            logger.warning(f"| ⚠️ Agent {agent_info.name} has no instance")
            continue
        
        # Create a wrapper function that calls the agent
        def create_agent_wrapper(agent=agent_instance, agent_name=agent_info.name):
            async def agent_wrapper(task: str, files: Optional[List[str]] = None, **kwargs) -> ActionResult:
                """Wrapper function to call agent.
                
                Args:
                    task: The task to complete
                    files: The files to attach to the task
                    
                Returns:
                    ActionResult: The result of the agent call
                """
                try:
                    # Call agent using __call__ method
                    result = await agent(task=task, files=files, **kwargs)
                    
                    # Convert AgentResponse to ActionResult
                    if isinstance(result, AgentResponse):
                        return ActionResult(
                            success=result.success,
                            message=result.message,
                            extra=result.extra.model_dump() if result.extra else None
                        )
                    elif isinstance(result, ActionResult):
                        return result
                    elif isinstance(result, dict):
                        return ActionResult(
                            success=result.get("success", True),
                            message=result.get("message", str(result)),
                            extra=result.get("extra")
                        )
                    else:
                        return ActionResult(
                            success=True,
                            message=str(result)
                        )
                except Exception as e:
                    return ActionResult(
                        success=False,
                        message=f"Error executing agent {agent_name}: {str(e)}"
                    )
            return agent_wrapper
        
        agent_wrapper_func = create_agent_wrapper()
        
        # Store type and args_schema in metadata if they exist
        action_metadata = agent_info.metadata.copy() if agent_info.metadata else {}
        action_metadata['type'] = agent_info.type
        if agent_info.args_schema:
            action_metadata['args_schema'] = agent_info.args_schema
        
        # Create ActionConfig - function_calling, text, and args_schema are computed automatically
        actions[agent_info.name] = ActionConfig(
            env_name=env_name,
            name=agent_info.name,
            description=agent_info.description,
            function=agent_wrapper_func,
            metadata=action_metadata
        )
    
    class ComposedEnvironment(Environment):
        name: str = Field(default=env_name, description="The name of the composed environment")
        type: str = Field(default="Composed Environment", description="The type of the composed environment")
        description: str = Field(default=env_description, description="The description of the composed environment")
        args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the composed environment")
        metadata: Dict[str, Any] = Field(default=metadata_, description="The metadata of the composed environment")
        
        # Store actions as class attribute
        _composed_actions = actions
        
        model_config = ConfigDict(
            arbitrary_types_allowed=True, 
            extra="allow"
        )
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Copy actions from class attribute to instance
            self.actions = self._composed_actions.copy()
            
        async def initialize(self) -> None:
            """Initialize the composed environment."""
            logger.info(f"| ✅ Composed environment {env_name} initialized")
        
        async def cleanup(self) -> None:
            """Cleanup the composed environment."""
            logger.info(f"| ✅ Composed environment {env_name} cleaned up")
        
        async def get_state(self) -> Dict[str, Any]:
            """Get the state of the composed environment."""
            return {
                "state": f"The state of the composed environment from {len(selected_agent_infos)} agents: {', '.join([a.name for a in selected_agent_infos])}"
            }
    
    return ComposedEnvironment


class A2ETransformer:
    """Transformer for converting ACP agents to ECP environments."""
    
    async def transform(self, request: A2ERequest) -> A2EResponse:
        """Convert ACP agents to ECP environments.
        
        This function takes multiple ACP agents and combines them into a single
        ECP environment where each agent becomes an action in the environment.
        
        Args:
            request (A2ERequest): A2ERequest instance with agent names to combine
            
        Returns:
            A2EResponse: A2EResponse with success status and message
        """
        try:
            logger.info("| 🔧 ACP to ECP transformation")
            
            # Step 1: Collect selected agent information
            selected_agent_infos = []
            for agent_name in request.agent_names:
                agent_info = await acp.get_info(agent_name)
                if agent_info:
                    selected_agent_infos.append(agent_info)
                else:
                    logger.warning(f"| ⚠️ Agent {agent_name} not found in ACP")
            
            if not selected_agent_infos:
                return A2EResponse(
                    success=False,
                    message="No valid agents found for transformation"
                )
            
            # Step 2: Use LLM to generate environment information
            class DynamicComposedArgs(BaseModel):
                name: str = Field(description="The name of the composed environment, the name should be a snake_case string.")
                description: str = Field(description="The description of the composed environment, the description should be a concise description of the environment.")
            
            agent_descriptions = "\n".join([f"- {a.name}: {a.description}" for a in selected_agent_infos])
            prompt = dedent(f"""
                You are a helpful assistant that composes an environment from a list of agents.
                
                The agents are:
                {agent_descriptions}
                
                Please compose an environment and give the name and description of the environment.
            """)
            
            from src.message.types import HumanMessage
            response = await model_manager(
                model="openrouter/gemini-3-flash-preview",
                messages=[HumanMessage(content=prompt)],
                response_format=DynamicComposedArgs
            )
            response = response.extra["parsed_model"]
            
            env_name = response.name
            env_description = response.description
            
            metadata_ = {
                "has_vision": False,
                "type": "Composed Environment",
                "additional_rules": {
                    "state": f"The state of the composed environment from {len(selected_agent_infos)} agents: {', '.join([a.name for a in selected_agent_infos])}"
                }
            }
            
            # Create environment class (actions are created inside)
            ComposedEnvironment = create_composed_environment_class(
                selected_agent_infos,
                env_name,
                env_description,
                metadata_
            )
            
            # Register the environment class
            await ecp.register(ComposedEnvironment, env_config_dict=None, override=True)
            
            logger.info(f"| ✅ A2E: Environment {env_name} created with {len(selected_agent_infos)} agents")
            
            return A2EResponse(
                success=True,
                message=f"Successfully created environment {env_name} with {len(selected_agent_infos)} agents"
            )
            
        except Exception as e:
            logger.error(f"| ❌ ACP to ECP transformation failed: {e}")
            return A2EResponse(
                success=False,
                message="ACP to ECP transformation failed: " + str(e)
            )
