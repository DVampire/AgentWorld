"""Transformation server for protocol conversions.

This server handles transformations between ECP, TCP, and ACP protocols.
"""

import asyncio
from typing import Any, List, Optional, Type, Dict
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field, ConfigDict

from src.config import config
from src.logger import logger
from src.tools.protocol.tool import WrappedTool
from src.tools.protocol.types import ToolInfo
from src.tools.protocol.server import tcp
from src.environments.protocol.environment import BaseEnvironment
from src.environments.protocol.types import ActionInfo, EnvironmentInfo
from src.environments.protocol.server import ecp
from src.agents.protocol.agent import BaseAgent
from src.agents.protocol.types import AgentInfo
from src.agents.protocol.server import acp
from src.agents import ToolCallingAgent, ThinkOutputBuilder
from src.infrastructures.models import model_manager
from src.utils import dedent
from src.transformation.protocol.types import (
    TransformationType,
    T2ERequest,
    T2EResponse,
    E2TRequest,
    E2TResponse,
    T2ARequest,
    T2AResponse,
    E2ARequest,
    E2AResponse,
    A2TRequest,
    A2TResponse,
    A2ERequest,
    A2EResponse,
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
            elif type == TransformationType.T2E.value:
                request = T2ERequest(
                    type=type,
                    tool_names=tool_names
                )
                result = await self._t2e(request)
            elif type == TransformationType.T2A.value:
                request = T2ARequest(
                    type=type,
                    tool_names=tool_names
                )
                result = await self._t2a(request)
            elif type == TransformationType.E2A.value:
                request = E2ARequest(
                    type=type,
                    env_names=env_names
                )
                result = await self._e2a(request)
            elif type == TransformationType.A2T.value:
                request = A2TRequest(
                    type=type,
                    agent_names=agent_names
                )
                result = await self._a2t(request)
            elif type == TransformationType.A2E.value:
                request = A2ERequest(
                    type=type,
                    agent_names=agent_names
                )
                result = await self._a2e(request)
            else:
                raise ValueError(f"Unknown transformation type: {type}")
            
            logger.info(f"| ✅ Transformation completed: {type}")
            return result
            
        except Exception as e:
            logger.error(f"| ❌ Transformation failed: {e}")
            return "Transformation failed: " + str(e)
        
    async def _t2e(self, request: T2ERequest) -> T2EResponse:
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
                tool_info = tcp.get_info(tool_name)
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
            
            class ComposedEnvironment(BaseEnvironment):
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

            env_info = EnvironmentInfo(
                name=env_name,
                type=env_type,
                rules="",
                description=env_description,
                args_schema=args_schema,
                actions={},
                cls=ComposedEnvironment,
                instance=None,
                metadata=metadata_
            )
            actions = {}
            for tool_info in selected_tool_infos:
                actions[tool_info.name] = ActionInfo(
                    env_name=env_name,
                    name=tool_info.name,
                    type=tool_info.type,
                    description=tool_info.description,
                    args_schema=tool_info.args_schema,
                    function=tool_info.instance._arun,
                    metadata=tool_info.metadata
                )
            env_info.actions = actions
            
            rules = ecp.genetate_rules(
                env_name,
                env_type,
                env_description,
                actions,
                metadata_.get('has_vision', False),
                metadata_.get('additional_rules', {})
            )
            env_info.rules = rules
            
            await ecp.register(env_info)
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
            
    async def _t2a(self, request: T2ARequest) -> T2AResponse:
        """Convert TCP tools to ACP agents.
        
        Args:
            request (T2ARequest): T2ARequest instance
            
        Returns:
            T2AResponse: T2AResponse
        """
        try:
            logger.info("| 🔧 TCP to ACP transformation")
            
            for tool_name in request.tool_names:
                tool_info = tcp.get_info(tool_name)
                if tool_info:
                    await tcp.register(tool_info)
                else:
                    logger.warning(f"| ⚠️ Tool {tool_name} not found in TCP")
                    
            return T2AResponse(
                success=True,
                message="TCP to ACP transformation completed",
            )
            
        except Exception as e:
            logger.error(f"| ❌ TCP to ACP transformation failed: {e}")
            return T2AResponse(
                success=False,
                message="TCP to ACP transformation failed: " + str(e)
            )
        
    
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
                    
                    await tcp.register(tool_info)
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
            
    async def _e2a(self, request: E2ARequest) -> E2AResponse:
        """Convert ECP environments to ACP agents.
        
        Args:
            request (E2ARequest): E2ARequest instance
            
        Returns:
            E2AResponse: E2AResponse
        """
        try:
            logger.info("| 🔧 ECP to ACP transformation")
            
            await self._e2t(E2TRequest(
                type=TransformationType.E2T.value,
                env_names=request.env_names
            ))
            
            selected_env_infos = []
            for env_name in request.env_names:
                env_info = ecp.get_info(env_name)
                if env_info:
                    selected_env_infos.append(env_info)
                else:
                    logger.warning(f"| ⚠️ Environment {env_name} not found in ECP")
                    
            if not selected_env_infos:
                return E2AResponse(
                    success=False,
                    message="No valid environments found for transformation"
                )
                
            selected_tool_infos = []
            for env_name in request.env_names:
                env_info = ecp.get_info(env_name)
                selected_tool_infos.extend(env_info.actions.values())
                
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
            
            env_descriptions = "\n".join([f"- {e.name}: {e.description}" for e in selected_env_infos])
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
                    
                    # Get args_schemas from ActionInfo objects
                    args_schemas = {}
                    for action_info in selected_tool_infos:
                        if hasattr(action_info, 'args_schema') and action_info.args_schema:
                            args_schemas[action_info.name] = action_info.args_schema
                    
                    self.think_output_builder = ThinkOutputBuilder()
                    self.think_output_builder.register(args_schemas)
                    self.ThinkOutput = self.think_output_builder.build()
                    
                    # Bind tools to model - get tools from TCP using action names
                    self.tools = []
                    for action_info in selected_tool_infos:
                        tool = tcp.get(action_info.name)
                        if tool:
                            self.tools.append(tool)
                    self.no_fc_model = self.model.bind_tools(tools=self.tools, tool_choice="none")
                    self.fc_model = self.model.bind_tools(tools=self.tools, tool_choice="any")
                    
                async def ainvoke(self, task: str, files: Optional[List[str]] = None) -> Any:
                    return await super().ainvoke(task, files)
                
                def invoke(self, task: str, files: Optional[List[str]] = None) -> Any:
                    return super().invoke(task, files)
                
            agent_info = AgentInfo(
                name=agent_name,
                type=agent_type,
                description=agent_description,
                args_schema=ComposedAgentInputArgs,
                metadata=metadata_,
                cls=ComposedAgent,
                instance=None
            )
            
            agent_info.instance = ComposedAgent(
                workdir=config.workdir,
            )
            
            await acp.register(agent_info)
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
            
    async def _a2t(self, request: A2TRequest) -> A2TResponse:
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
                agent_info = acp.get_info(agent_name)
                
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
                
                def make_wrapped_func(agent_info):
                    if asyncio.iscoroutinefunction(agent_info.instance.ainvoke):
                        async def _async_action_wrapper(**kwargs):
                            return await agent_info.instance.ainvoke(**kwargs)
                        return _async_action_wrapper
                    else:
                        def _sync_action_wrapper(**kwargs):
                            return agent_info.instance.invoke(**kwargs)
                        return _sync_action_wrapper
                
                tool = StructuredTool(
                    name=agent_info.name,
                    description=agent_info.description,
                    args_schema=agent_info.args_schema,
                    func=make_wrapped_func(agent_info),
                    coroutine=make_wrapped_func(agent_info) if asyncio.iscoroutinefunction(agent_info.instance.ainvoke) else None,
                    metadata=agent_info.metadata
                )
                tool = WrappedTool(tool=tool)
                tool_info = ToolInfo(
                    name=agent_info.name,
                    type=agent_info.type,
                    description=agent_info.description,
                    args_schema=agent_info.args_schema,
                    metadata=agent_info.metadata,
                    cls=WrappedTool,
                    instance=tool
                )
                tool_info.instance = tool
                
                await tcp.register(tool_info)
                logger.info(f"| ✅ ACP to TCP transformation completed: {tool.name}")
                
            logger.info(f"| ✅ ACP to TCP transformation completed: {len(selected_agent_infos)} tools")
            
        except Exception as e:
            logger.error(f"| ❌ ACP to TCP transformation failed: {e}")
            return A2TResponse(
                success=False,
                message="ACP to TCP transformation failed: " + str(e)
            )
            
    async def _a2e(self, request: A2ERequest) -> A2EResponse:
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
                agent_info = acp.get_info(agent_name)
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
            
            class ComposedEnvironment(BaseEnvironment):
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

            env_info = EnvironmentInfo(
                name=env_name,
                type=env_type,
                rules="",
                description=env_description,
                args_schema=args_schema,
                actions={},
                cls=ComposedEnvironment,
                instance=None,
                metadata=metadata_
            )
            
            # Step 3: Create actions for each agent
            actions = {}
            for agent_info in selected_agent_infos:
                actions[agent_info.name] = ActionInfo(
                    env_name=env_name,
                    name=agent_info.name,
                    type=agent_info.type,
                    description=agent_info.description,
                    args_schema=agent_info.args_schema,
                    function=agent_info.instance.ainvoke,
                    metadata=agent_info.metadata
                )
            env_info.actions = actions
            
            # Step 4: Generate rules
            rules = ecp.genetate_rules(
                env_name,
                env_type,
                env_description,
                actions,
                metadata_.get('has_vision', False),
                metadata_.get('additional_rules', {})
            )
            env_info.rules = rules
            
            # Step 5: Register to ECP
            await ecp.register(env_info)
            
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
        

transformation = TransformationServer()