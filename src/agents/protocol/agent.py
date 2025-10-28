"""Base agent class for multi-agent system."""
import pandas as pd
from typing import List, Optional, Type, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from src.logger import logger
from src.infrastructures.models import model_manager
from src.agents.prompts import PromptManager
from src.infrastructures.memory import MemoryManager
from src.agents.protocol.types import InputArgs
from src.utils import get_file_info, dedent
from src.infrastructures.memory import SessionInfo, EventType
from src.tools.protocol import tcp
from src.environments.protocol import ecp

def format_actions(actions: List[BaseModel]) -> str:
    """Format actions as a Markdown table using pandas."""
    rows = []
    for action in actions:
        if isinstance(action.args, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in action.args.items())
        else:
            args_str = str(action.args)

        rows.append({
            "Action": action.name,
            "Args": args_str,
            "Output": action.output if action.output is not None else None
        })
    
    df = pd.DataFrame(rows)
    
    if df["Output"].isna().all():
        df = df.drop(columns=["Output"])
    else:
        df["Output"] = df["Output"].fillna("None")
    
    return df.to_markdown(index=True)

class ThinkOutputBuilder:
    def __init__(self):
        self.schemas: Dict[str, type[BaseModel]] = {}

    def register(self, schema: Dict[str, type[BaseModel]]):
        """Register new args schema"""
        self.schemas.update(schema)
        return self  # Support chaining

    def build(self):
        """Generate Action and ThinkOutput models"""

        # -------- Dynamically generate Action --------
        schemas = self.schemas
        ActionArgs = Union[tuple(schemas.values())]

        class Action(BaseModel):
            name: str = Field(description="The name of the action.")
            args: ActionArgs = Field(description="The arguments of the action.")
            output: Optional[str] = Field(default=None, description="The output of the action.")
            
            def __str__(self):
                return f"Action: {self.name}\nArgs: {self.args}\nOutput: {self.output}\n"
            
            def __repr__(self):
                return self.__str__()

        # -------- Dynamically generate ThinkOutput --------
        class ThinkOutput(BaseModel):
            thinking: str = Field(description="A structured <think>-style reasoning block.")
            evaluation_previous_goal: str = Field(description="One-sentence analysis of your last action.")
            memory: str = Field(description="1-3 sentences of specific memory.")
            next_goal: str = Field(description="State the next immediate goals and actions.")
            action: List[Action] = Field(
                description='[{"name": "action_name", "args": {...}}, ...]'
            )

            def __str__(self):
                return (
                    f"Thinking: {self.thinking}\n"
                    f"Evaluation of Previous Goal: {self.evaluation_previous_goal}\n"
                    f"Memory: {self.memory}\n"
                    f"Next Goal: {self.next_goal}\n"
                    f"Action:\n{format_actions(self.action)}\n"
                )
            
            def __repr__(self):
                return self.__str__()

        return ThinkOutput

class BaseAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(description="The name of the agent.")
    type: str = Field(description="The type of the agent.")
    description: str = Field(description="The description of the agent.")
    args_schema: Type[InputArgs] = Field(description="The args schema of the agent.")
    metadata: Dict[str, Any] = Field(description="The metadata of the agent.")
    
    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        type: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[InputArgs]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_modules: Optional[Dict[str, Any]] = None,
        memory_config: Dict[str, Any] = None,
        max_tools: int = 10,
        max_steps: int = 20,
        review_steps: int = 5,
        log_max_length: int = 1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Set default values
        self.name = name or self.name
        self.type = type or self.type
        self.description = description or self.description
        self.args_schema = args_schema or self.args_schema
        self.metadata = metadata or self.metadata
        
        self.workdir = workdir
        logger.info(f"| 📁 Agent working directory: {self.workdir}")
        
        self.prompt_manager = PromptManager(prompt_name=prompt_name)
        self.prompt_modules = prompt_modules or {}
        self.memory_manager = MemoryManager(memory_config=memory_config)
        
        # Setup model
        self.model = self._setup_model(model_name)
        
        # Setup steps
        self.max_steps = max_steps if max_steps>0 else int(1e8)
        self.max_tools = max_tools
        if self.max_tools > 0:
            self.prompt_modules["max_tools"] = self.max_tools
        self.review_steps = review_steps
        self.step_number = 0
        self.log_max_length = log_max_length

    def _setup_model(self, model_name: Optional[str]):
        """Setup the language model."""
        if model_name:
            # Get model from ModelManager
            model = model_manager.get(model_name)
            if model:
                return model
            else:
                logger.warning(f"Warning: Model '{model_name}' not found in model_manager")
        
        # Fallback to default model
        default_model = model_manager.get("gpt-4.1")
        if default_model:
            return default_model
        else:
            raise RuntimeError("No model available")
    
    def __str__(self):
        return f"Agent(name={self.name}, model={self.model_name}, prompt_name={self.prompt_name})"
    
    def __repr__(self):
        return self.__str__()
        
    async def _extract_file_content(self, file: str) -> str:
        """Extract file information."""
        
        info = get_file_info(file)
        
        # Extract file content
        file_content = await tcp.ainvoke("mdify", input={"file_path": file, "output_format": "markdown"})
        
        # Use LLM to summarize the file content
        system_prompt = "You are a helpful assistant that summarizes file content."
        
        user_prompt = dedent(f"""
            Summarize the following file content as 1-3 sentences:
            {file_content}
        """)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        summary = await self.no_fc_model.ainvoke(messages)
        
        info["content"] = file_content
        info["summary"] = summary.content
        
        return info
    
    async def _generate_enhanced_task(self, task: str, files: List[str]) -> str:
        """Generate enhanced task."""
        
        attach_files_string = "\n".join([f"File: {file['path']}\nSummary: {file['summary']}" for file in files])
        
        enhanced_task = dedent(f"""
        - Task:
        {task}
        - Attach files:
        {attach_files_string}
        """)
        return enhanced_task
    
    async def _generate_session_info(self, task: str) -> SessionInfo:
        """Use the llm to generate a session id."""
        structured_llm = self.model.with_structured_output(
            SessionInfo,
            method="function_calling",
            include_raw=False
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant that generates a session info for agent {self.name}."),
            ("user", 
             dedent(f"""
                    <intro>
                    1. The session ID should be a unique identifier for the session that concisely describes the task in snake_case.
                    2. The session description should provide a concise description of the task.
                    </intro>
                    <task>
                    {task}
                    </task>"""
                )
             )
        ])

        chain = prompt | structured_llm
        result: SessionInfo = chain.invoke({"task": task})
        
        timestamp = datetime.now().isoformat()
        
        session_id = f"{self.name}_{timestamp}"
        description = result.description
        
        return SessionInfo(session_id=session_id, description=description)
    
    async def _get_agent_context(self, task: str) -> Dict[str, Any]:
        """Get the agent context."""
        
        task = f"<task>{task}</task>"
        
        step_info_description = f'Step {self.step_number + 1} of {self.max_steps} max possible steps\n'
        time_str = datetime.now().isoformat()
        step_info_description += f'Current date and time: {time_str}'
        step_info = dedent(f"""
            <step_info>
            {step_info_description}
            </step_info>
        """)
        
        state = await self.memory_manager.get_state(n=self.review_steps)
        
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
            elif event.event_type == EventType.ACTION_STEP:
                agent_history += f"Evaluation of Previous Step: {event.data['evaluation_previous_goal']}\n"
                agent_history += f"Memory: {event.data['memory']}\n"
                agent_history += f"Next Goal: {event.data['next_goal']}\n"
                agent_history += f"Action Results: {event.data['action']}\n"
            agent_history += "\n"
            agent_history += f"</step_{event.step_number}>\n"
        agent_history += "</agent_history>"
        
        memory = "<memory>"
        if len(summaries) > 0:
            memory += dedent(f"""
                <summaries>
                {chr(10).join([str(summary) for summary in summaries])}
                </summaries>
            """)
        else:
            memory += "<summaries>[Current summaries are empty.]</summaries>\n"
        if len(insights) > 0:
            memory += dedent(f"""
                <insights>
                {chr(10).join([str(insight) for insight in insights])}
                </insights>
            """)
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
        todo_tool = tcp.get("todo")
        todo_contents = todo_tool.get_todo_content()
        return todo_contents
        
    async def _get_environment_context(self) -> Dict[str, Any]:
        """Get the environment state."""
        environment_context = "<environment_context>"
        for env_name in ecp.list():
            rule_string = ecp.get_info(env_name).rules
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
                    state_string += f"\n<img src={screenshot.screenshot_path} alt={screenshot.screenshot_description}/>"
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
        
        tool_list = [tcp.to_string(tool) for tool in tcp.list()]
        tool_list_string = "\n".join(tool_list)
        
        tool_context += dedent(f"""
        <tool_list>
        {tool_list_string}
        </tool_list>
        """)
        
        tool_context += "</tool_context>"
        return {
            "tool_context": tool_context,
        }
        
    async def _get_messages(self, task: str) -> List[BaseMessage]:
        
        system_modules = self.prompt_modules
        system_message = self.prompt_manager.get_system_message(modules=system_modules, reload=False)
        
        agent_message_modules = self.prompt_modules
        agent_message_modules.update(await self._get_agent_context(task))
        agent_message_modules.update(await self._get_environment_context())
        agent_message_modules.update(await self._get_tool_context())
        agent_message = self.prompt_manager.get_agent_message(modules=agent_message_modules, reload=True)
        
        messages = [
            system_message,
            agent_message,
        ]
        
        return messages

    async def ainvoke(self,  task: str, files: Optional[List[str]] = None):
        """Run the agent. This method should be implemented by the child classes."""
        raise NotImplementedError("Run method is not implemented by the child class")
    
    def invoke(self,  task: str, files: Optional[List[str]] = None):
        """Run the agent. This method should be implemented by the child classes."""
        raise NotImplementedError("Run method is not implemented by the child class")
