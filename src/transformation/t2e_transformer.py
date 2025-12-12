"""Tool to Environment (T2E) Transformer.

Converts TCP tools to ECP environments.
"""

from typing import Any, Dict, Type
from pydantic import BaseModel, Field, ConfigDict

from src.config import config
from src.logger import logger
from src.tool.server import tcp
from src.environment.types import Environment, EnvironmentConfig, ActionConfig
from src.environment.server import ecp
from src.model import model_manager
from src.utils import dedent
from src.transformation.types import T2ERequest, T2EResponse


class T2ETransformer:
    """Transformer for converting TCP tools to ECP environments."""
    
    async def transform(self, request: T2ERequest) -> T2EResponse:
        """Convert TCP tools to ECP environments.
        
        This function takes multiple TCP tools and combines them into a single
        ECP environment where each tool becomes an action in the environment.
        
        Args:
            request (T2ERequest): T2ERequest instance with tool names to combine
            
        Returns:
            T2EResponse: T2EResponse with success status and message
        """
        try:
            logger.info("| 🔧 TCP to ECP transformation")
            
            selected_tool_infos = []
            for tool_name in request.tool_names:
                tool_info = await tcp.get_info(tool_name)
                if tool_info:
                    selected_tool_infos.append(tool_info)
                else:
                    logger.warning(f"| ⚠️ Tool {tool_name} not found in TCP")
            
            if not selected_tool_infos:
                return T2EResponse(
                    success=False,
                    message="No valid tools found for transformation"
                )
            
            class DynamicComposedArgs(BaseModel):
                name: str = Field(description="The name of the composed environment, the name should be a snake_case string.")
                description: str = Field(description="The description of the composed environment, the description should be a concise description of the environment.")
                
            model = model_manager.get("gpt-4.1")
            structured_llm = model.with_structured_output(
                DynamicComposedArgs,
                method="function_calling",
                include_raw=False
            )
            
            tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in selected_tool_infos])
            prompt = dedent(f"""
                You are a helpful assistant that composes an environment from a list of tools.
                
                The tools are:
                {tool_descriptions}
                
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
                    "state": f"The state of the composed environment from {len(selected_tool_infos)} tools: {', '.join([t.name for t in selected_tool_infos])}"
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
                        "state": f"The state of the composed environment from {len(selected_tool_infos)} tools: {', '.join([t.name for t in selected_tool_infos])}"
                    }

            actions = {}
            for tool_info in selected_tool_infos:
                tool_instance = await tcp.get(tool_info.name)
                # Use _arun method if available, otherwise use __call__
                if hasattr(tool_instance, '_arun'):
                    func = tool_instance._arun
                elif hasattr(tool_instance, '__call__'):
                    func = tool_instance.__call__
                else:
                    logger.warning(f"| ⚠️ Tool {tool_info.name} has no callable method")
                    continue
                    
                # Store type and args_schema in metadata if they exist
                action_metadata = tool_info.metadata.copy() if tool_info.metadata else {}
                if hasattr(tool_info, 'type') and tool_info.type:
                    action_metadata['type'] = tool_info.type
                if hasattr(tool_info, 'args_schema') and tool_info.args_schema:
                    action_metadata['args_schema'] = tool_info.args_schema
                
                actions[tool_info.name] = ActionConfig(
                    env_name=env_name,
                    name=tool_info.name,
                    description=tool_info.description,
                    function=func,
                    metadata=action_metadata
                )
            
            # Generate rules using context manager's method
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
            
            await ecp.register(env_config)
            logger.info(f"| ✅ Composed environment {env_name} registered to ECP")
            
            return T2EResponse(
                success=True,
                message=f"Successfully created environment {env_name} with {len(selected_tool_infos)} tools"
            )
            
        except Exception as e:
            logger.error(f"| ❌ TCP to ECP transformation failed: {e}")
            return T2EResponse(
                success=False,
                message="TCP to ECP transformation failed: " + str(e)
            )
