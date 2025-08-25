"""Tool calling agent implementation with manual agent logic."""

import json
import re
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage

from src.agents.base_agent import BaseAgent, ThinkOutput
from src.registry import AGENTS
from src.logger import logger
from src.config import config
from src.memory import MemoryManager
from src.filesystem import FileSystem

@AGENTS.register_module(force=True)
class ToolCallingAgent(BaseAgent):
    """Tool calling agent implementation with manual agent logic."""
    
    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        model: Optional[BaseLanguageModel] = None,
        prompt_name: Optional[str] = None,
        tools: Optional[List[Union[str, BaseTool]]] = None,
        max_iterations: int = 10,
        verbose: bool = True,
        memory_manager: Optional[MemoryManager] = None,
        file_system: Optional[FileSystem] = None,
        **kwargs
    ):
        # Set default prompt name for tool calling
        if not prompt_name:
            prompt_name = "tool_calling"
        
        super().__init__(
            name=name,
            model_name=model_name,
            model=model,
            prompt_name=prompt_name,
            tools=tools,
            memory_manager=memory_manager,
            file_system=file_system,
            **kwargs)
        
        self.max_iterations = max_iterations
        self.verbose = verbose
        
    async def _think_and_action(self, messages: List[BaseMessage]):
        """Think and action."""
        structured_llm = self.no_fc_model.with_structured_output(
            ThinkOutput,
            method="function_calling",
            include_raw=False
        )
        think_output = structured_llm.invoke(messages)

        action = think_output.action
        for item in action:
            tool_name = item.name
            tool_args = item.args
            tool_args = tool_args.model_dump()
            
            results = await self.tool_manager.execute_tool(tool_name, args=tool_args)
        
    
          
    async def run(self, 
                  task: str,
                  ):
        """Run the tool calling agent."""
        
        session_info = self._generate_session_info(task)
        session_id = session_info.session_id
        description = session_info.description
        
        # Start session
        self.memory_manager.start_session(session_id, description)
        
        # Add task start event
        task_id = str(uuid.uuid4())
        self.memory_manager.add_event(step_number=self.step_number, 
                                      event_type="task_start", 
                                      data=dict(task=task),
                                      agent_name=self.name,
                                      task_id=task_id
                                      )
        self.step_number += 1
        
        messages = self._get_messages(task)
        
        await self._think_and_action(messages)