"""Base agent class for multi-agent system."""
from abc import ABC
import pandas as pd
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from src.tools.protocol.tool import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from src.infrastructures.models import model_manager
from src.tools.protocol import tcp
from src.agents.prompts import PromptManager
from src.logger import logger
from src.infrastructures.memory import MemoryManager, SessionInfo, EventType
from src.config import config
from src.environments.protocol import ecp
from src.utils import dedent

class Action(BaseModel):
    """Action."""
    name: str = Field(description="The name of the action.")
    args: Union[*tcp.args_schemas()] = Field(description="The arguments of the action.")
    output: Optional[str] = Field(description="The output of the action. This is only provided if the action is completed.", default=None)

    def __str__(self):
        """String representation of the action."""
        str = f"Action: {self.name}\n"
        str += f"Args: {self.args}\n"
        str += f"Output: {self.output}\n"
        return str
    
    def __repr__(self):
        return self.__str__()
    
def format_actions(actions: List[Action]) -> str:
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
    
class ThinkOutput(BaseModel):
    """Think output."""
    thinking: str = Field(description="A structured <think>-style reasoning block that applies the <reasoning_rules> provided above.")
    evaluation_previous_goal: str = Field(description="One-sentence analysis of your last action. Clearly state success, failure, or uncertain.")
    memory: str = Field(description="1-3 sentences of specific memory of this step and overall progress. You should put here everything that will help you track progress in future steps.")
    next_goal: str = Field(description="State the next immediate goals and actions to achieve it, in one clear sentence.")
    action: List[Action] = Field(description='[{"name": "action_name", "args": {// action-specific parameters}}, // ... more actions in sequence], the action should be in the <available_actions>.')
    
    def __str__(self):
        """String representation of the think output."""
        str = f"Thinking: {self.thinking}\n"
        str += f"Evaluation of Previous Goal: {self.evaluation_previous_goal}\n"
        str += f"Memory: {self.memory}\n"
        str += f"Next Goal: {self.next_goal}\n"
        str += f"Action: \n{format_actions(self.action)}\n"
        return str
    
    def __repr__(self):
        return self.__str__()

class BaseAgent(ABC):
    """Base class for all agents in the multi-agent system."""
    
    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        model: Optional[BaseLanguageModel] = None,
        prompt_name: Optional[str] = None,
        tools: Optional[List[Union[str, BaseTool]]] = None,
        memory_manager: Optional[MemoryManager] = None,
        env_names: Optional[List[str]] = None,
        max_steps: int = 20,
        review_steps: int = 5,
        log_max_length: int = 1000,
        **kwargs
    ):
        self.name = name
        self.model_name = model_name
        self.prompt_name = prompt_name
        
        self.workdir = getattr(config, 'workdir', 'workdir')
        logger.info(f"| 📁 Agent working directory: {self.workdir}")
        
        self.model_manager = model_manager
        self.prompt_manager = PromptManager(prompt_name=prompt_name)
        self.memory_manager = memory_manager or MemoryManager()
        self.env_names = env_names or []
        
        # Setup model
        self.model = self._setup_model(model_name, model)
        
        # Setup tools
        self.tools = self._setup_tools(tools)
        # Bind tools to model
        self.no_fc_model = self.model.bind_tools(tools=self.tools, tool_choice="none")
        self.fc_model = self.model.bind_tools(tools=self.tools, tool_choice="any")
        
        # Setup steps
        self.max_steps = max_steps if max_steps>0 else int(1e8)
        self.review_steps = review_steps
        self.step_number = 0
        self.log_max_length = log_max_length

    def _setup_model(self, model_name: Optional[str], model: Optional[BaseLanguageModel]):
        """Setup the language model."""
        if model:
            # Use provided LLM directly
            return model
        elif model_name:
            # Get model from ModelManager
            model = self.model_manager.get(model_name)
            if model:
                return model
            else:
                logger.warning(f"Warning: Model '{model_name}' not found in ModelManager")
        
        # Fallback to default model
        default_model = self.model_manager.get("gpt-4.1")
        if default_model:
            return default_model
        else:
            raise RuntimeError("No model available")
    
    def _setup_tools(self, tools: Optional[List[Union[str, BaseTool]]]):
        """Setup and validate tools for the agent."""
        if not tools:
            return
        
        valid_tools = []
        for tool in tools:
            if isinstance(tool, str):
                # Get tool by name from ToolManager
                tool_obj = tcp.get(tool)
                if tool_obj:
                    if hasattr(tool_obj, 'name') and hasattr(tool_obj, 'description'):
                        valid_tools.append(tool_obj)
                        logger.debug(f"| ✅ Tool validated: {tool_obj.name}")
                    else:
                        logger.warning(f"| ⚠️ Invalid tool found: {tool_obj}")
                else:
                    logger.warning(f"Warning: Tool '{tool}' not found in ToolManager")
            elif isinstance(tool, BaseTool):
                # Add tool directly if valid
                if hasattr(tool, 'name') and hasattr(tool, 'description'):
                    valid_tools.append(tool)
                    logger.debug(f"| ✅ Tool validated: {tool.name}")
                else:
                    logger.warning(f"| ⚠️ Invalid tool found: {tool}")
        
        if len(valid_tools) != len(tools):
            logger.warning(f"| ⚠️ Some tools were invalid. Valid tools: {len(valid_tools)}/{len(tools)}")
        
        return valid_tools
    
    def __str__(self):
        return f"Agent(name={self.name}, model={self.model_name}, prompt_name={self.prompt_name})"
    
    def __repr__(self):
        return self.__str__()
    
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
    
    async def _get_agent_history(self) -> Dict[str, Any]:
        """Get the agent history."""
        state = await self.memory_manager.get_state(n=self.review_steps)
        
        events = state["events"]
        summaries = state["summaries"]
        insights = state["insights"]
        
        agent_history = ""
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
        
        agent_history += dedent(f"""
            <summaries>
            {chr(10).join([str(summary) for summary in summaries])}
            </summaries>
            <insights>
            {chr(10).join([str(insight) for insight in insights])}
            </insights>
        """)
        
        return {
            "agent_history": agent_history,
        }
    
    async def _get_todo_contents(self) -> str:
        """Get the todo contents."""
        todo_tool = tcp.get("todo")
        todo_contents = todo_tool.get_todo_content()
        return todo_contents   
    
    async def _get_agent_state(self, task: str) -> Dict[str, Any]:
        """Get the agent state."""
        step_info_description = f'Step {self.step_number + 1} of {self.max_steps} max possible steps\n'
        time_str = datetime.now().isoformat()
        step_info_description += f'Current date and time: {time_str}'
        
        available_actions_description = [tcp.to_string(tool) for tool in self.tools]
        available_actions_description = "\n".join(available_actions_description)
        
        todo_contents = await self._get_todo_contents()
        
        return {
            "task": task,
            "step_info": step_info_description,
            "available_actions": available_actions_description,
            "todo_contents": todo_contents,
        }
        
    async def _get_environment_state(self) -> Dict[str, Any]:
        """Get the environment state."""
        environment_state = ""
        for env_name in self.env_names:
            state = await ecp.get_state(env_name)
            state_string = state.get("state", "")
            environment_state += f"{state_string}\n"
        return {
            "environment_state": environment_state,
        }
        
    async def _get_messages(self, task: str) -> List[BaseMessage]:
        
        system_input_variables = {}
        environment_rules = ""
        for env_name in self.env_names:
            environment_rules += f"{ecp.get_info(env_name).rules}\n"
        system_input_variables.update(dict(
            environment_rules=environment_rules,
        ))
        system_message = self.prompt_manager.get_system_message(system_input_variables)
        
        agent_input_variables = {}
        agent_history = await self._get_agent_history()
        agent_state = await self._get_agent_state(task)
        environment_state = await self._get_environment_state()
        agent_input_variables.update(agent_history)
        agent_input_variables.update(agent_state)
        agent_input_variables.update(environment_state)
        agent_message = self.prompt_manager.get_agent_message(agent_input_variables)
        
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
