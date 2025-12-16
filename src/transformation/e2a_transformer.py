"""Environment to Agent (E2A) Transformer.

Converts ECP environments to ACP agents.
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field, ConfigDict

from src.config import config
from src.logger import logger
from src.environment.server import ecp
from src.agent.server import acp
from src.agent import ToolCallingAgent
from src.model import model_manager
from src.utils import dedent
from src.transformation.types import E2ARequest, E2AResponse, E2TRequest, TransformationType


def create_composed_agent_class(selected_env_configs, agent_name: str, agent_description: str, metadata_: Dict[str, Any]):
    """Create a composed agent class from multiple environments.
    
    Args:
        selected_env_configs: List of environment configurations
        agent_name: Name of the composed agent
        agent_description: Description of the composed agent
        metadata_: Metadata for the agent
        
    Returns:
        Type[ToolCallingAgent]: The composed agent class
    """
    # Collect all action configs from all environments
    selected_action_configs = []
    for env_config in selected_env_configs:
        if env_config and env_config.actions:
            selected_action_configs.extend(env_config.actions.values())
    
    class ComposedAgentInputArgs(BaseModel):
        task: str = Field(description="The task to complete.")
        files: Optional[List[str]] = Field(default=None, description="The files to attach to the task.")
    
    class ComposedAgent(ToolCallingAgent):
        name: str = Field(default=agent_name, description="The name of the composed agent")
        type: str = Field(default="Composed Agent", description="The type of the composed agent")
        description: str = Field(default=agent_description, description="The description of the composed agent")
        args_schema: Type[ComposedAgentInputArgs] = Field(default=ComposedAgentInputArgs, description="The args schema of the composed agent.")
        metadata: Dict[str, Any] = Field(default=metadata_, description="The metadata of the composed agent")
        
        # Store action configs as class attribute
        _composed_action_configs = selected_action_configs
        
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
            
            # Copy action configs from class attribute to instance
            self._action_configs = self._composed_action_configs.copy()
            self._args_schemas_initialized = False
        
        async def initialize(self):
            """Initialize the agent and build ThinkOutput with args_schemas."""
            # Call parent initialize first
            await super().initialize()
            
            # Get args_schemas from ActionConfig properties (computed automatically)
            if not self._args_schemas_initialized:
                args_schemas = {}
                for action_config in self._action_configs:
                    # Get args_schema directly from ActionConfig property (automatically computed)
                    try:
                        if action_config.function is not None:
                            args_schema = action_config.args_schema
                            if args_schema:
                                args_schemas[action_config.name] = args_schema
                    except Exception as e:
                        logger.warning(f"| ⚠️ Could not get args_schema for action {action_config.name}: {e}")
                        continue
                
                # Register additional args_schemas if any
                if args_schemas:
                    self.think_output_builder.register(args_schemas)
                    self.ThinkOutput = self.think_output_builder.build()
                
                self._args_schemas_initialized = True
    
    return ComposedAgent


class E2ATransformer:
    """Transformer for converting ECP environments to ACP agents."""
    
    def __init__(self, e2t_transformer):
        """Initialize E2A transformer.
        
        Args:
            e2t_transformer: E2T transformer instance for converting environments to tools first
        """
        self.e2t_transformer = e2t_transformer
    
    async def transform(self, request: E2ARequest) -> E2AResponse:
        """Convert ECP environments to ACP agents.
        
        This function takes multiple ECP environments and combines them into a single
        ACP agent that can use the tools converted from these environments.
        
        Args:
            request (E2ARequest): E2ARequest instance with environment names to combine
            
        Returns:
            E2AResponse: E2AResponse with success status and message
        """
        try:
            logger.info("| 🔧 ECP to ACP transformation")
            
            # Step 1: First convert environments to tools
            await self.e2t_transformer.transform(E2TRequest(
                type=TransformationType.E2T.value,
                env_names=request.env_names
            ))
            
            # Step 2: Collect selected environment information
            selected_env_configs = []
            for env_name in request.env_names:
                env_config = await ecp.get_info(env_name)
                if env_config:
                    selected_env_configs.append(env_config)
                else:
                    logger.warning(f"| ⚠️ Environment {env_name} not found in ECP")
                    
            if not selected_env_configs:
                return E2AResponse(
                    success=False,
                    message="No valid environments found for transformation"
                )
            
            # Step 3: Use LLM to generate agent information
            class DynamicComposedArgs(BaseModel):
                name: str = Field(description="The name of the composed agent, the name should be a snake_case string.")
                description: str = Field(description="The description of the composed agent, the description should be a concise description of the agent.")
            
            env_descriptions = "\n".join([f"- {e.name}: {e.description}" for e in selected_env_configs])
            prompt = dedent(f"""
                You are a helpful assistant that composes an agent from a list of environments.
                
                The environments are:
                {env_descriptions}
                
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
            
            # Step 4: Create agent class (action configs are collected inside)
            ComposedAgent = create_composed_agent_class(
                selected_env_configs,
                agent_name,
                agent_description,
                metadata_
            )
            
            # Step 5: Register the agent class
            await acp.register(ComposedAgent, config={"workdir": config.workdir}, override=True)
            
            logger.info(f"| ✅ E2A: Agent {agent_name} created with {len(selected_env_configs)} environments")
            
            return E2AResponse(
                success=True,
                message=f"Successfully created agent {agent_name} with {len(selected_env_configs)} environments"
            )
                    
        except Exception as e:
            logger.error(f"| ❌ ECP to ACP transformation failed: {e}")
            return E2AResponse(
                success=False,
                message="ECP to ACP transformation failed: " + str(e)
            )
