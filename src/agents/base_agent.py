"""Base agent class for multi-agent system."""

from abc import ABC
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from regex import D
from langchain_core.utils.function_calling import convert_to_openai_function

from src.models import model_manager
from src.tools import tool_manager
from src.prompts import PromptManager
from src.logger import logger
from src.memory import MemoryManager, SessionInfo, EventType
from src.filesystem import FileSystem
from src.config import config

class Action(BaseModel):
    """Action."""
    name: str = Field(description="The name of the action.")
    args: Union[*tool_manager.list_tool_args_schemas()] = Field(description="The arguments of the action.")
    
def format_actions(actions: List[Action]) -> str:
    """Format actions."""
    return json.dumps([action.model_dump() for action in actions], ensure_ascii=False)
    
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
        str += f"Action: {format_actions(self.action)}\n"
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
        file_system: Optional[FileSystem] = None,
        max_steps: int = 20,
        review_steps: int = 5,
        **kwargs
    ):
        self.name = name
        self.model_name = model_name
        self.prompt_name = prompt_name
        
        self.workdir = getattr(config, 'workdir', 'workdir')
        logger.info(f"| 📁 Agent working directory: {self.workdir}")
        
        self.model_manager = model_manager
        self.tool_manager = tool_manager
        self.prompt_manager = PromptManager(prompt_name=prompt_name)
        self.memory_manager = memory_manager or MemoryManager()
        self.file_system = file_system or FileSystem(base_dir=self.workdir)
        
        # Setup model
        self.model = self._setup_model(model_name, model)
        
        # Setup tools
        self.tools = self._setup_tools(tools)
        # Bind tools to model
        self.no_fc_model = self.model.bind_tools(tools=self.tools, tool_choice="none")
        self.fc_model = self.model.bind_tools(tools=self.tools, tool_choice="any")
        
        # Setup steps
        self.max_steps = max_steps
        self.review_steps = review_steps
        self.step_number = 0
    
    def _setup_model(self, model_name: Optional[str], model: Optional[BaseLanguageModel]):
        """Setup the language model."""
        if model:
            # Use provided LLM directly
            return model
        elif model_name:
            # Get model from ModelManager
            model = self.model_manager.get_model(model_name)
            if model:
                return model
            else:
                logger.warning(f"Warning: Model '{model_name}' not found in ModelManager")
        
        # Fallback to default model
        default_model = self.model_manager.get_model("gpt-4.1")
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
                tool_obj = self.tool_manager.get_tool(tool)
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
    
    def _generate_session_info(self, task: str) -> SessionInfo:
        """Use the llm to generate a session id."""
        structured_llm = self.model.with_structured_output(
            SessionInfo,
            method="function_calling",
            include_raw=False
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant that generates a session info for agent {self.name}."),
            ("user", """<intro>
1. The session ID should be a unique identifier for the session that concisely describes the task in snake_case.
2. The session description should provide a concise description of the task.
</intro>
<task>
{task}
</task>""")
        ])

        chain = prompt | structured_llm
        result: SessionInfo = chain.invoke({"task": task})
        
        timestamp = datetime.now().isoformat()
        
        session_id = f"{self.name}_{timestamp}"
        description = result.description
        
        return SessionInfo(session_id=session_id, description=description)
    
    def _get_agent_history(self) -> Dict[str, Any]:
        """Get the agent history."""
        events = self.memory_manager.get_events(
            num = self.review_steps
        )
        
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
            agent_history += f"<step_{self.step_number}>\n"
        
        return {
            "agent_history": agent_history,
        }
        
    def _get_agent_state(self, task: str) -> Dict[str, Any]:
        """Get the agent state."""
        step_info_description = f'Step {self.step_number + 1} of {self.max_steps} max possible steps\n'
        time_str = datetime.now().isoformat()
        step_info_description += f'Current date and time: {time_str}'
        
        todo_contents = self.file_system.get_todo_contents() if self.file_system else ''
        if not len(todo_contents):
            todo_contents = '[Current todo.md is empty, fill it with your plan when applicable]'
        
        file_system_contents = self.file_system.describe() if self.file_system else 'No file system available'
        
        available_actions_description = [
            convert_to_openai_function(tool) for tool in self.tools
        ]
        available_actions_description = json.dumps(available_actions_description)
        
        return {
            "task": task,
            "step_info": step_info_description,
            "todo_contents": todo_contents,
            "file_system": file_system_contents,
            "available_actions": available_actions_description,
        }
        
    def _get_messages(self, task: str) -> List[BaseMessage]:
        input_variables = {}
        agent_history = self._get_agent_history()
        agent_state = self._get_agent_state(task)
        
        input_variables.update(agent_history)
        input_variables.update(agent_state)
        
        messages = []
        messages.append(self.prompt_manager.get_system_message())
        messages.append(self.prompt_manager.get_agent_message(input_variables))
        
        return messages
    
    async def run(self, task: str):
        """Run the agent. This method should be implemented by the child classes."""
        raise NotImplementedError("Run method is not implemented by the child class")
