"""Agent Context Protocol (ACP) Types

Core type definitions for the Agent Context Protocol and common Agent
abstractions, aligned with the design of `src.tool.types`.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union, get_type_hints

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.config import config
from src.environment.server import ecp
from src.logger import logger
from src.memory import EventType, SessionInfo, memory_manager
from src.message.types import HumanMessage, Message, SystemMessage
from src.model import model_manager
from src.prompt import prompt_manager
from src.tool.server import tcp
from src.utils import (
    PYTHON_TYPE_FIELD,
    annotation_to_types,
    build_args_schema,
    build_function_calling,
    build_text_representation,
    default_parameters_schema,
    dedent,
    get_file_info,
    parse_docstring_descriptions,
)

class InputArgs(BaseModel):
    task: str = Field(description="The task to complete.")
    files: Optional[List[str]] = Field(default=None, description="The files to attach to the task.")

class ACPErrorCode(Enum):
    """ACP error codes."""
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    AGENT_NOT_FOUND = -32001

class ACPError(BaseModel):
    """ACP error structure."""
    code: ACPErrorCode
    message: str
    data: Optional[Dict[str, Any]] = None

class ACPRequest(BaseModel):
    """ACP request structure."""
    id: Union[str, int] = Field(default_factory=lambda: str(uuid.uuid4()))
    method: str
    params: Optional[Dict[str, Any]] = None

class ACPResponse(BaseModel):
    """ACP response structure."""
    id: Union[str, int]
    result: Optional[Dict[str, Any]] = None
    error: Optional[ACPError] = None

class AgentConfig(BaseModel):
    """Agent configuration for registration, similar to `ToolConfig`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str
    description: str
    version: str = Field(default="1.0.0", description="Version of the agent")

    cls: Optional[Any] = None
    config: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="The initialization configuration of the agent",
    )
    instance: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    code: Optional[str] = Field(
        default=None,
        description="Source code for dynamically generated agent classes (used when cls cannot be imported from a module)",
    )

    function_calling: Optional[Dict[str, Any]] = Field(
        default=None, description="Default function calling representation"
    )
    text: Optional[str] = Field(
        default=None, description="Default text representation of the agent"
    )
    args_schema: Optional[Type[BaseModel]] = Field(
        default=None, description="Default args schema (BaseModel type)"
    )

    def __str__(self) -> str:
        return (
            f"AgentConfig(name={self.name}, "
            f"description={self.description})"
        )

    def __repr__(self) -> str:
        return self.__str__()


def format_tools(tools: List[BaseModel]) -> str:
    """Format tools as a Markdown table using pandas."""
    rows = []
    for tool in tools:
        if isinstance(tool.args, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in tool.args.items())
        else:
            args_str = str(tool.args)

        rows.append({
            "Tool": tool.name,
            "Args": args_str,
            "Output": tool.output if tool.output is not None else None
        })
    
    df = pd.DataFrame(rows)
    
    if df["Output"].isna().all():
        df = df.drop(columns=["Output"])
    else:
        df["Output"] = df["Output"].fillna("None")
    
    return df.to_markdown(index=True)

class ThinkOutputBuilder:
    """Builder used by agents to dynamically construct ThinkOutput schemas."""

    def __init__(self) -> None:
        self.schemas: Dict[str, Type[BaseModel]] = {}

    def register(self, schema: Dict[str, Type[BaseModel]]) -> "ThinkOutputBuilder":
        """Register new args schema."""
        self.schemas.update(schema)
        return self  # Support chaining

    def build(self) -> Type[BaseModel]:
        """Generate Tool and ThinkOutput models."""

        # -------- Dynamically generate Tool --------
        schemas = self.schemas
        ToolArgs = Union[tuple(schemas.values())]  # type: ignore[type-arg]

        class Tool(BaseModel):
            name: str = Field(description="The name of the tool.")
            args: ToolArgs = Field(description="The arguments of the tool.")
            output: Optional[str] = Field(
                default=None, description="The output of the tool."
            )

            def __str__(self) -> str:
                return f"Tool: {self.name}\nArgs: {self.args}\nOutput: {self.output}\n"

            def __repr__(self) -> str:
                return self.__str__()

        # -------- Dynamically generate ThinkOutput --------
        class ThinkOutput(BaseModel):
            thinking: str = Field(
                description="A structured <think>-style reasoning block."
            )
            evaluation_previous_goal: str = Field(
                description="One-sentence analysis of your last tool call."
            )
            memory: str = Field(description="1-3 sentences of specific memory.")
            next_goal: str = Field(
                description="State the next immediate goals and tool calls."
            )
            tool: List[Tool] = Field(
                description='[{"name": "tool_name", "args": {...}}, ...]'
            )

            def __str__(self) -> str:
                return (
                    f"Thinking: {self.thinking}\n"
                    f"Evaluation of Previous Goal: {self.evaluation_previous_goal}\n"
                    f"Memory: {self.memory}\n"
                    f"Next Goal: {self.next_goal}\n"
                    f"Tool:\n{format_tools(self.tool)}\n"
                )

            def __repr__(self) -> str:
                return self.__str__()

        return ThinkOutput


class Agent(BaseModel):
    """Base class for all agents, mirroring the design of `Tool`."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(description="The name of the agent.")
    description: str = Field(description="The description of the agent.")
    metadata: Dict[str, Any] = Field(description="The metadata of the agent.")

    @staticmethod
    def default_parameters_schema() -> Dict[str, Any]:
        """Default empty-parameters schema, same as tools."""
        return default_parameters_schema()

    @property
    def parameter_schema(self) -> Dict[str, Any]:
        """JSONSchema-like schema built from the agent's `__call__` signature."""
        schema = self._build_parameter_schema()
        return dict(schema)

    @property
    def function_calling(self) -> Dict[str, Any]:
        """OpenAI-compatible function-calling representation for the agent."""
        schema = self.parameter_schema
        return build_function_calling(self.name, self.description, schema)

    @property
    def args_schema(self) -> Type[BaseModel]:
        """Return a BaseModel type for the agent's input parameters."""
        schema = self.parameter_schema
        return build_args_schema(self.name, schema)

    @property
    def text(self) -> str:
        """Human-readable text representation of the agent and its arguments."""
        schema = self.parameter_schema
        return build_text_representation(
            self.name,
            self.description,
            schema,
            entity_type="Agent",
        )

    def _build_parameter_schema(self) -> Dict[str, Any]:
        """Build parameter schema from `__call__` signature and docstring."""
        import inspect

        try:
            signature = inspect.signature(self.__class__.__call__)
        except (TypeError, ValueError):
            return self.default_parameters_schema()

        # Get type hints
        try:
            hints = get_type_hints(self.__class__.__call__)
        except Exception:
            hints = {}

        # Get docstring for descriptions
        docstring = inspect.getdoc(self.__class__.__call__) or ""
        doc_descriptions = parse_docstring_descriptions(docstring)

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for name, param in signature.parameters.items():
            if name == "self":
                continue

            # Skip VAR_KEYWORD (**kwargs) and VAR_POSITIONAL (*args) parameters
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            # Get type annotation
            annotation = hints.get(name, param.annotation)
            json_type, python_type = annotation_to_types(annotation)

            # Determine if required
            is_required = param.default is inspect._empty

            # Build schema
            field_schema: Dict[str, Any] = {
                "type": json_type,
                "description": doc_descriptions.get(name, ""),
            }
            field_schema[PYTHON_TYPE_FIELD] = python_type

            if not is_required:
                field_schema["default"] = param.default

            properties[name] = field_schema
            if is_required:
                required.append(name)

        if not properties:
            return self.default_parameters_schema()

        result: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            result["required"] = required
        return result

    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_modules: Optional[Dict[str, Any]] = None,
        memory_name: Optional[str] = None,
        max_tools: int = 10,
        max_steps: int = 20,
        review_steps: int = 5,
        log_max_length: int = 1000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        # Set default values
        self.name = name or self.name
        self.description = description or self.description
        self.metadata = metadata or self.metadata

        # Set working directory
        self.workdir = workdir

        # Set prompt name and modules
        self.prompt_name = prompt_name
        self.memory_name = memory_name
        self.model_name = model_name

        # Setup steps
        self.prompt_modules = prompt_modules or {}
        self.max_steps = max_steps if max_steps > 0 else int(1e8)
        self.max_tools = max_tools
        if self.max_tools > 0:
            self.prompt_modules["max_tools"] = self.max_tools
        self.prompt_modules["workdir"] = self.workdir
        self.review_steps = review_steps
        self.step_number = 0
        self.log_max_length = log_max_length

    async def initialize(self) -> None:
        """Initialize the agent."""
        logger.info(f"| 📁 Agent working directory: {self.workdir}")
        # Setup think output builder
        self.think_output_builder = ThinkOutputBuilder()

        self.tool_names = config.tool_names
        # Get all tools asynchronously
        tools = await asyncio.gather(*[tcp.get(tool_name) for tool_name in self.tool_names])
        args_schema = {
            tool_name: tool.args_schema for tool_name, tool in zip(self.tool_names, tools)
        }
        self.think_output_builder.register(args_schema)
        self.ThinkOutput = self.think_output_builder.build()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, model={self.model_name}, prompt_name={self.prompt_name})"

    def __repr__(self) -> str:
        return self.__str__()

    async def _extract_file_content(self, file: str) -> Dict[str, Any]:
        """Extract file information and a short summary."""

        info = get_file_info(file)

        # Extract file content
        input_payload = {
            "name": "mdify",
            "input": {
                "file_path": file,
                "output_format": "markdown",
            },
        }
        tool_response = await tcp(**input_payload)
        file_content = tool_response.message

        # Use LLM to summarize the file content
        system_prompt = "You are a helpful assistant that summarizes file content."

        user_prompt = dedent(
            f"""
            Summarize the following file content as 1-3 sentences:
            {file_content}
        """
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        model_response = await model_manager(model=self.model_name, messages=messages)

        info["content"] = file_content
        info["summary"] = model_response.message

        return info

    async def _generate_enhanced_task(self, task: str, files: List[Dict[str, Any]]) -> str:
        """Generate enhanced task with attached file summaries."""

        attach_files_string = "\n".join(
            [f"File: {file['path']}\nSummary: {file['summary']}" for file in files]
        )

        enhanced_task = dedent(
            f"""
        - Task:
        {task}
        - Attach files:
        {attach_files_string}
        """
        )
        return enhanced_task

    async def _generate_session_info(self, task: str) -> SessionInfo:
        """Generate a session id based on task without extra LLM calls."""

        # Generate a simple session_id based on task content (first 50 chars) and timestamp
        task_hash = hashlib.md5(task.encode("utf-8")).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_id = f"{self.name}_{timestamp}_{task_hash}"

        # Generate a simple description (first 100 chars of task)
        description = task[:100] + "..." if len(task) > 100 else task

        logger.info(f"| ✅ Session info generated: {session_id}")
        return SessionInfo(session_id=session_id, description=description)

    async def _get_agent_context(self, task: str) -> Dict[str, Any]:
        """Get the agent context."""

        task = f"<task>{task}</task>"

        step_info_description = (
            f"Step {self.step_number + 1} of {self.max_steps} max possible steps\n"
        )
        time_str = datetime.now().isoformat()
        step_info_description += f"Current date and time: {time_str}"
        step_info = dedent(f"""
            <step_info>
            {step_info_description}
            </step_info>
        """)

        state = await memory_manager.get_state(
            name=self.memory_name,
            n=self.review_steps,
        )

        events = state["events"]
        summaries = state["summaries"]
        insights = state["insights"]

        agent_history = "<agent_history>"
        for event in events:
            agent_history += f"<step_{event.step_number}>\n"
            if event.event_type == EventType.TASK_START:
                agent_history += f"Task Start: {event.data['task']}\n"
            elif event.event_type == EventType.TASK_END:
                agent_history += f"Task End: {event.data['result']}\n"
            elif event.event_type == EventType.TOOL_STEP:
                agent_history += f"Evaluation of Previous Step: {event.data['evaluation_previous_goal']}\n"
                agent_history += f"Memory: {event.data['memory']}\n"
                agent_history += f"Next Goal: {event.data['next_goal']}\n"
                agent_history += f"Tool Results: {event.data.get('tool', '')}\n"
            agent_history += "\n"
            agent_history += f"</step_{event.step_number}>\n"
        agent_history += "</agent_history>"

        memory = "<memory>"
        if len(summaries) > 0:
            memory += dedent(
                f"""
                <summaries>
                {chr(10).join([str(summary) for summary in summaries])}
                </summaries>
            """
            )
        else:
            memory += "<summaries>[Current summaries are empty.]</summaries>\n"
        if len(insights) > 0:
            memory += dedent(
                f"""
                <insights>
                {chr(10).join([str(insight) for insight in insights])}
                </insights>
            """
            )
        else:
            memory += "<insights>[Current insights are empty.]</insights>\n"
        memory += "</memory>"

        todo = "<todo>"
        todo_contents = await self._get_todo_contents()
        todo += todo_contents
        todo += "</todo>"

        agent_context = dedent(f"""
            <agent_context>
            {task}
            {step_info}
            {agent_history}
            {memory}
            {todo}
            </agent_context>
        """)

        return {
            "agent_context": agent_context,
        }

    async def _get_todo_contents(self) -> str:
        """Get the todo contents."""
        todo_tool = await tcp.get("todo")
        todo_contents = todo_tool.get_todo_content()
        return todo_contents

    async def _get_environment_context(self) -> Dict[str, Any]:
        """Get the environment state."""
        environment_context = "<environment_context>"
        # Only iterate over environments specified in config, not all registered environments
        for env_name in config.env_names:
            env_info = await ecp.get_info(env_name)
            rule_string = env_info.rules
            rule_string = dedent(f"""
                <rules>
                {rule_string}
                </rules>
            """)

            env_state = await ecp.get_state(env_name)
            state_string = "<state>"
            state_string += env_state["state"]
            extra = env_state["extra"]

            if "screenshots" in extra:
                for screenshot in extra["screenshots"]:
                    state_string += (
                        f"\n<img src={screenshot.screenshot_path} "
                        f"alt={screenshot.screenshot_description}/>"
                    )
            state_string += "</state>"

            environment_context += dedent(f"""
                <{env_name}>
                {rule_string}
                {state_string}
                </{env_name}>
            """)

        environment_context += "</environment_context>"
        return {
            "environment_context": environment_context,
        }

    async def _get_tool_context(self) -> Dict[str, Any]:
        """Get the tool context."""
        tool_context = "<tool_context>"

        tool_context += dedent(f"""
            <available_tools>
            {await tcp.get_contract()}
            </available_tools>
        """)

        tool_context += "</tool_context>"
        return {
            "tool_context": tool_context,
        }

    async def _get_messages(self, task: str) -> List[Message]:
        """Build system+agent messages using prompt templates and context."""

        system_modules = self.prompt_modules.copy()
        # Infer prompt name from agent's prompt_name
        if self.prompt_name:
            system_prompt_name = f"{self.prompt_name}_system_prompt"
            agent_message_prompt_name = f"{self.prompt_name}_agent_message_prompt"
        else:
            system_prompt_name = "tool_calling_system_prompt"
            agent_message_prompt_name = "tool_calling_agent_message_prompt"

        system_message = await prompt_manager.get_system_message(
            prompt_name=system_prompt_name,
            modules=system_modules,
            reload=False,
        )

        agent_message_modules = self.prompt_modules.copy()
        agent_message_modules.update(await self._get_agent_context(task))
        agent_message_modules.update(await self._get_environment_context())
        agent_message_modules.update(await self._get_tool_context())
        agent_message = await prompt_manager.get_agent_message(
            prompt_name=agent_message_prompt_name,
            modules=agent_message_modules,
            reload=True,
        )

        messages = [
            system_message,
            agent_message,
        ]

        return messages

    async def __all__(self, task: str, files: Optional[List[str]] = None):
        """Run the agent. This method should be implemented by the child classes."""
        raise NotImplementedError("__all__ method is not implemented by the child class")


__all__ = [
    "InputArgs",
    "ACPErrorCode",
    "ACPError",
    "ACPRequest",
    "ACPResponse",
    "AgentConfig",
    "ThinkOutputBuilder",
    "Agent",
]
