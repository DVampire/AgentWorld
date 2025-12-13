"""Environment to Agent (E2A) Transformer.

Converts ECP environments to ACP agents.
"""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field, ConfigDict

from src.config import config
from src.logger import logger
from src.tool.server import tcp
from src.environment.server import ecp
from src.agent.types import ThinkOutputBuilder, AgentConfig
from src.agent.server import acp
from src.agent import ToolCallingAgent
from src.model import model_manager
from src.utils import dedent
from src.transformation.types import E2ARequest, E2AResponse, E2TRequest, TransformationType


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
        
        Args:
            request (E2ARequest): E2ARequest instance
            
        Returns:
            E2AResponse: E2AResponse
        """
        try:
            logger.info("| 🔧 ECP to ACP transformation")
            
            # First convert environments to tools
            await self.e2t_transformer.transform(E2TRequest(
                type=TransformationType.E2T.value,
                env_names=request.env_names
            ))
            
            selected_env_configs = []
            for env_name in request.env_names:
                env_config = ecp.get_info(env_name)
                if env_config:
                    selected_env_configs.append(env_config)
                else:
                    logger.warning(f"| ⚠️ Environment {env_name} not found in ECP")
                    
            if not selected_env_configs:
                return E2AResponse(
                    success=False,
                    message="No valid environments found for transformation"
                )
                
            selected_action_configs = []
            for env_name in request.env_names:
                env_config = ecp.get_info(env_name)
                selected_action_configs.extend(env_config.actions.values())
                
            class ComposedAgentArgs(BaseModel):
                name: str = Field(description="The name of the composed agent, the name should be a snake_case string.")
                description: str = Field(description="The description of the composed agent, the description should be a concise description of the agent.")
                
            class ComposedAgentInputArgs(BaseModel):
                task: str = Field(description="The task to complete.")
                files: Optional[List[str]] = Field(default=None, description="The files to attach to the task.")
                
            model = model_manager.get("gpt-4.1")
            structured_llm = model.with_structured_output(
                ComposedAgentArgs,
                method="function_calling",
                include_raw=False
            )
            
            env_descriptions = "\n".join([f"- {e.name}: {e.description}" for e in selected_env_configs])
            prompt = dedent(f"""
                You are a helpful assistant that composes an agent from a list of environments.
                
                The environments are:
                {env_descriptions}
                
                Please compose an agent and give the name and description of the agent.
            """)
            
            response = await structured_llm.ainvoke(prompt)
            
            agent_name = response.name
            agent_type = "Composed Agent"
            agent_description = response.description
            metadata_ = {}
            
            class ComposedAgent(ToolCallingAgent):
                name: str = Field(default=agent_name, description="The name of the composed agent")
                type: str = Field(default=agent_type, description="The type of the composed agent")
                description: str = Field(default=agent_description, description="The description of the composed agent")
                args_schema: Type[ComposedAgentInputArgs] = Field(default=ComposedAgentInputArgs, description="The args schema of the composed agent.")
                metadata: Dict[str, Any] = Field(default={}, description="The metadata of the composed agent")
                
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
                    
                    # Store action configs for reference (tools are already registered in TCP from e2t step)
                    self._action_configs = selected_action_configs
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
                
            agent_config = AgentConfig(
                name=agent_name,
                type=agent_type,
                description=agent_description,
                args_schema=ComposedAgentInputArgs,
                metadata=metadata_,
                cls=ComposedAgent,
                instance=None
            )
            
            agent_config.instance = ComposedAgent(
                workdir=config.workdir,
            )
            
            # Initialize the agent to set up args_schemas
            await agent_config.instance.initialize()
            
            await acp.register(agent_config)
            logger.info(f"| ✅ ECP to ACP transformation completed: {agent_name}")
            
            return E2AResponse(
                success=True,
                message=f"ECP to ACP transformation completed: {agent_name}",
            )
                    
        except Exception as e:
            logger.error(f"| ❌ ECP to ACP transformation failed: {e}")
            return E2AResponse(
                success=False,
                message="ECP to ACP transformation failed: " + str(e)
            )
