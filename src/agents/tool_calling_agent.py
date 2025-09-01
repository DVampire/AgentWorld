"""Tool calling agent implementation with manual agent logic."""

import uuid
from typing import List, Optional, Union
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage

from src.agents.base_agent import BaseAgent, ThinkOutput
from src.registry import AGENTS
from src.logger import logger
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
        
    async def _think_and_action(self, messages: List[BaseMessage], task_id: str):
        """Think and action for one step."""
        # Get structured output for thinking
        structured_llm = self.no_fc_model.with_structured_output(
            ThinkOutput,
            method="function_calling",
            include_raw=False
        )
        
        done = False
        final_result = None
        
        try:
            think_output = structured_llm.invoke(messages)
            
            thinking = think_output.thinking
            evaluation_previous_goal = think_output.evaluation_previous_goal
            memory = think_output.memory
            next_goal = think_output.next_goal
            actions = think_output.action
            
            logger.info(f"| 💭 Thinking: {thinking[:100]}...")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 🔧 Actions to execute: {len(actions)}")
            
            # Execute actions sequentially
            action_results = []
            
            for i, action in enumerate(actions):
                logger.info(f"| 📝 Action {i+1}/{len(actions)}: {action.name}")
                
                # Execute the tool
                tool_name = action.name
                tool_args = action.args.model_dump()
                
                tool_result = await self.tool_manager.execute_tool(tool_name, args=tool_args)
                
                logger.info(f"| ✅ Action {i+1} completed successfully")
                logger.info(f"| 📄 Results: {str(tool_result)[:200]}...")
                
                # Update action with result
                action_dict = action.model_dump()
                action_dict["output"] = tool_result
                action_results.append(action_dict)
                    
                if tool_name == "done":
                    done = True
                    final_result = tool_result
                    break
            
            event_data = {
                "thinking": thinking,
                "evaluation_previous_goal": evaluation_previous_goal,
                "memory": memory,
                "next_goal": next_goal,
                "action": action_results
            }
            self.memory_manager.add_event(
                step_number=self.step_number,
                event_type="action_step",
                data=event_data,
                agent_name=self.name,
                task_id=task_id
            )
            self.step_number += 1
            
            if done:
                self.memory_manager.add_event(
                    step_number=self.step_number,
                    event_type="task_end",
                    data=dict(result=final_result),
                    agent_name=self.name,
                    task_id=task_id
                )
            
        except Exception as e:
            logger.error(f"| Error in thinking and action step: {e}")
        
        return done, final_result
        
          
    async def run(self, task: str):
        """Run the tool calling agent with loop."""
        logger.info(f"| 🚀 Starting ToolCallingAgent: {task}")
        
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
        
        # Initialize messages
        messages = self._get_messages(task)
        
        # Main loop
        iteration = 0
        done = False
        final_result = None
        
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"| 🔄 Iteration {iteration}/{self.max_iterations}")
            
            # Execute one step
            done, final_result = await self._think_and_action(messages, task_id)
            self.step_number += 1
            
            messages = self._get_messages(task)
            
            if done:
                break
        
        # Handle max iterations reached
        if iteration >= self.max_iterations:
            logger.warning(f"| 🛑 Reached max iterations ({self.max_iterations}), stopping...")
            final_result = "Reached maximum number of iterations"
        
        # Add task end event
        self.memory_manager.add_event(
            step_number=self.step_number,
            event_type="task_end",
            data=dict(result=final_result),
            agent_name=self.name,
            task_id=task_id
        )
        
        logger.info(f"| ✅ Agent completed after {iteration}/{self.max_iterations} iterations")
        
        return final_result