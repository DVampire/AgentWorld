"""Tool calling agent implementation with manual agent logic."""

import asyncio
from typing import List, Optional, Union
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from datetime import datetime

from src.agents.protocol.agent import BaseAgent, ThinkOutput
from src.logger import logger
from src.infrastructures.memory import MemoryManager
from src.utils import get_file_info, dedent
from src.agents.protocol import acp
from src.tools.protocol import tcp
from src.tools.protocol.types import ToolResponse

@acp.agent(
    name="tool_calling",
    type="Agent",
    description="A tool calling agent that can call tools to complete tasks.",
    metadata={}
)
class ToolCallingAgent(BaseAgent):
    """Tool calling agent implementation with manual agent logic."""
    
    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        model: Optional[BaseLanguageModel] = None,
        prompt_name: Optional[str] = None,
        tools: Optional[List[Union[str, BaseTool]]] = None,
        memory_manager: Optional[MemoryManager] = None,
        env_names: Optional[List[str]] = None,
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
            env_names=env_names,
            **kwargs)
        
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
            think_output = await structured_llm.ainvoke(messages)
            
            thinking = think_output.thinking
            evaluation_previous_goal = think_output.evaluation_previous_goal
            memory = think_output.memory
            next_goal = think_output.next_goal
            actions = think_output.action
            
            logger.info(f"| 💭 Thinking: {thinking[:self.log_max_length]}...")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 🔧 Actions to execute: {len(actions)}")
            
            # Execute actions sequentially
            action_results = []
            
            for i, action in enumerate(actions):
                logger.info(f"| 📝 Action {i+1}/{len(actions)}: {action.name}")
                
                # Execute the tool
                tool_name = action.name
                tool_args = action.args.model_dump()
                
                logger.info(f"| 📝 Action Name: {tool_name}, Args: {tool_args}")
                
                tool_result = await tcp.ainvoke(tool_name, input=tool_args)
                if isinstance(tool_result, ToolResponse):
                    tool_result = tool_result.content
                else:
                    tool_result = str(tool_result)
                
                logger.info(f"| ✅ Action {i+1} completed successfully")
                logger.info(f"| 📄 Results: {str(tool_result)[:self.log_max_length]}...")
                
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
            await self.memory_manager.add_event(
                step_number=self.step_number,
                event_type="action_step",
                data=event_data,
                agent_name=self.name,
                task_id=task_id
            )
            self.step_number += 1
            
            if done:
                await self.memory_manager.add_event(
                    step_number=self.step_number,
                    event_type="task_end",
                    data=dict(result=final_result),
                    agent_name=self.name,
                    task_id=task_id
                )
            
        except Exception as e:
            logger.error(f"| Error in thinking and action step: {e}")
        
        return done, final_result
    
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
        
    async def ainvoke(self, 
                  task: str, 
                  files: Optional[List[str]] = None,
                  ):
        """Run the tool calling agent with loop."""
        logger.info(f"| 🚀 Starting ToolCallingAgent: {task}")
        
        if files:
            logger.info(f"| 📂 Attached files: {files}")
            files = await asyncio.gather(*[self._extract_file_content(file) for file in files])
            enhanced_task = await self._generate_enhanced_task(task, files)
        else:
            enhanced_task = task
        
        session_info = await self._generate_session_info(enhanced_task)
        session_id = session_info.session_id
        description = session_info.description
        
        # Start session
        await self.memory_manager.start_session(session_id, description)
        
        # Add task start event
        task_id = "task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        await self.memory_manager.add_event(step_number=self.step_number, 
                                      event_type="task_start", 
                                      data=dict(task=enhanced_task),
                                      agent_name=self.name,
                                      task_id=task_id
                                      )
        
        # Initialize messages
        messages = await self._get_messages(enhanced_task)
        
        # Main loop
        step_number = 0
        done = False
        final_result = None
        
        while step_number < self.max_steps:
            step_number += 1
            logger.info(f"| 🔄 Step {step_number}/{self.max_steps}")
            
            # Execute one step
            done, final_result = await self._think_and_action(messages, task_id)
            self.step_number += 1
            
            messages = await self._get_messages(enhanced_task)
            
            if done:
                break
        
        # Handle max steps reached
        if step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping...")
            final_result = "Reached maximum number of steps"
        
        # Add task end event
        await self.memory_manager.add_event(
            step_number=self.step_number,
            event_type="task_end",
            data=dict(result=final_result),
            agent_name=self.name,
            task_id=task_id
        )
        
        # End session
        await self.memory_manager.end_session(session_id=session_id)
        
        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")
        
        return final_result