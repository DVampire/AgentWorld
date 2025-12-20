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


def create_composed_environment_class(selected_tool_infos, tool_instances_map, env_name: str, env_description: str, metadata_: Dict[str, Any]):
    """Create a composed environment class from multiple tools.
    
    Args:
        selected_tool_infos: List of tool configurations
        tool_instances_map: Dictionary mapping tool names to tool instances
        env_name: Name of the composed environment
        env_description: Description of the composed environment
        metadata_: Metadata for the environment
        
    Returns:
        Type[Environment]: The composed environment class
    """
    # Create actions for each tool
    actions = {}
    for tool_info in selected_tool_infos:
        # Get tool instance from map
        tool_instance = tool_instances_map.get(tool_info.name)
        if tool_instance is None:
            logger.warning(f"| ⚠️ Tool {tool_info.name} has no instance")
            continue
        
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
        
        # Create ActionConfig - function_calling, text, and args_schema are computed automatically
        actions[tool_info.name] = ActionConfig(
            env_name=env_name,
            name=tool_info.name,
            description=tool_info.description,
            function=func,
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
                "state": f"The state of the composed environment from {len(selected_tool_infos)} tools: {', '.join([t.name for t in selected_tool_infos])}"
            }
    
    return ComposedEnvironment


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
            
            # Step 1: Collect selected tool information
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
            
            # Step 2: Use LLM to generate environment information
            class DynamicComposedArgs(BaseModel):
                name: str = Field(description="The name of the composed environment, the name should be a snake_case string.")
                description: str = Field(description="The description of the composed environment, the description should be a concise description of the environment.")
            
            tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in selected_tool_infos])
            prompt = dedent(f"""
                You are a helpful assistant that composes an environment from a list of tools.
                
                The tools are:
                {tool_descriptions}
                
                Please compose an environment and give the name and description of the environment.
            """)
            
            from src.message.types import HumanMessage
            response = await model_manager(
                model="gpt-4.1",
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
                    "state": f"The state of the composed environment from {len(selected_tool_infos)} tools: {', '.join([t.name for t in selected_tool_infos])}"
                }
            }
            
            # Step 3: Get tool instances
            tool_instances_map = {}
            for tool_info in selected_tool_infos:
                tool_instance = tool_info.instance
                if tool_instance is None:
                    # Try to get from tcp if instance is not available
                    try:
                        tool_instance = await tcp.get(tool_info.name)
                    except Exception as e:
                        logger.warning(f"| ⚠️ Tool {tool_info.name} has no instance and cannot be retrieved: {e}")
                        continue
                if tool_instance is not None:
                    tool_instances_map[tool_info.name] = tool_instance
            
            # Step 4: Create environment class (actions are created inside)
            ComposedEnvironment = create_composed_environment_class(
                selected_tool_infos,
                tool_instances_map,
                env_name,
                env_description,
                metadata_
            )
            
            # Step 5: Register the environment class
            await ecp.register(ComposedEnvironment, env_config_dict=None, override=True)
            
            logger.info(f"| ✅ T2E: Environment {env_name} created with {len(selected_tool_infos)} tools")
            
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
