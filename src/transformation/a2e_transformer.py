"""Agent to Environment (A2E) Transformer.

Converts ACP agents to ECP environments.
"""

from typing import Any, Dict, Type
from pydantic import BaseModel, Field, ConfigDict

from src.logger import logger
from src.environment.types import Environment, EnvironmentConfig, ActionConfig
from src.environment.server import ecp
from src.agent.server import acp
from src.model import model_manager
from src.utils import dedent
from src.transformation.types import A2ERequest, A2EResponse


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
                
            model = model_manager.get("gpt-4.1")
            structured_llm = model.with_structured_output(
                DynamicComposedArgs,
                method="function_calling",
                include_raw=False
            )
            
            agent_descriptions = "\n".join([f"- {a.name}: {a.description}" for a in selected_agent_infos])
            prompt = dedent(f"""
                You are a helpful assistant that composes an environment from a list of agents.
                
                The agents are:
                {agent_descriptions}
                
                Please compose an environment and give the name and description of the environment.
            """)
            
            response = await structured_llm.ainvoke(prompt)
            
            env_name = response.name
            env_type = "Composed Environment"
            env_description = response.description
            args_schema = None
            metadata_ = {
                "has_vision": False,
                "additional_rules": {
                    "state": f"The state of the composed environment from {len(selected_agent_infos)} agents: {', '.join([a.name for a in selected_agent_infos])}"
                }
            }
            
            class ComposedEnvironment(Environment):
                name: str = Field(default=env_name, description="The name of the composed environment")
                type: str = Field(default=env_type, description="The type of the composed environment")
                description: str = Field(default=env_description, description="The description of the composed environment")
                args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the composed environment")
                metadata: Dict[str, Any] = Field(default=metadata_, description="The metadata of the composed environment")
                
                model_config = ConfigDict(
                    arbitrary_types_allowed=True, 
                    extra="allow"
                )
                
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    
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

            # Step 3: Create actions for each agent
            actions = {}
            for agent_info in selected_agent_infos:
                # Store type and args_schema in metadata if they exist
                action_metadata = agent_info.metadata.copy() if agent_info.metadata else {}
                action_metadata['type'] = agent_info.type
                if agent_info.args_schema:
                    action_metadata['args_schema'] = agent_info.args_schema
                
                actions[agent_info.name] = ActionConfig(
                    env_name=env_name,
                    name=agent_info.name,
                    description=agent_info.description,
                    function=agent_info.instance.ainvoke,
                    metadata=action_metadata
                )
            
            # Step 4: Generate rules using context manager's method
            from src.environment.context import EnvironmentContextManager
            context_manager = EnvironmentContextManager()
            rules = context_manager._generate_rules_from_metadata(
                env_name,
                env_description,
                actions,
                metadata_.get('has_vision', False),
                metadata_.get('additional_rules', {})
            )
            
            # Store type in metadata
            metadata_['type'] = env_type
            if args_schema:
                metadata_['args_schema'] = args_schema
            
            env_config = EnvironmentConfig(
                name=env_name,
                rules=rules,
                description=env_description,
                actions=actions,
                cls=ComposedEnvironment,
                config={},
                instance=None,
                metadata=metadata_
            )
            
            # Step 5: Register to ECP
            await ecp.register(env_config)
            
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
