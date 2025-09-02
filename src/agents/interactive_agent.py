"""Interactive agent implementation with user interaction."""

import uuid
from typing import List, Optional, Union, Dict, Any
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage

from src.agents.base_agent import BaseAgent, ThinkOutput
from src.registry import AGENTS
from src.logger import logger
from src.memory import MemoryManager
from src.filesystem import FileSystem

@AGENTS.register_module(force=True)
class InteractiveAgent(BaseAgent):
    """Interactive agent with user interaction capabilities."""
    
    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        model: Optional[BaseLanguageModel] = None,
        prompt_name: Optional[str] = None,
        tools: Optional[List[Union[str, BaseTool]]] = None,
        max_iterations: int = 50,
        memory_manager: Optional[MemoryManager] = None,
        file_system: Optional[FileSystem] = None,
        interactive_mode: bool = True,
        auto_continue: bool = False,
        **kwargs
    ):
        # Set default prompt name for interactive agent
        if not prompt_name:
            prompt_name = "interactive"
        
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
        self.interactive_mode = interactive_mode
        self.auto_continue = auto_continue
        
    async def _think_and_action(self, messages: List[BaseMessage], task_id: str):
        """Think and action for one step with interactive capabilities."""
        # Get structured output for thinking
        structured_llm = self.no_fc_model.with_structured_output(
            ThinkOutput,
            method="function_calling",
            include_raw=False
        )
        
        done = False
        final_result = None
        
        try:
            # Display thinking status
            logger.info("| 🤔 Thinking...")
            think_output = await structured_llm.ainvoke(messages)
            
            thinking = think_output.thinking
            evaluation_previous_goal = think_output.evaluation_previous_goal
            memory = think_output.memory
            next_goal = think_output.next_goal
            actions = think_output.action
            
            # Display current status
            await self._display_status(thinking, next_goal, actions)
            
            # Execute actions sequentially
            action_results = []
            
            for i, action in enumerate(actions):
                logger.info(f"| 📝 Action {i+1}/{len(actions)}: {action.name}")
                
                # Check if this is a done action
                if action.name == "done":
                    logger.info("| 🏁 Agent called 'done' action")
                    done = True
                    final_result = action.args.get('text', 'Task completed')
                    break
                
                # Execute the tool
                tool_name = action.name
                tool_args = action.args.model_dump()
                
                try:
                    tool_result = await self.tool_manager.execute_tool(tool_name, args=tool_args)
                    
                    logger.info(f"| ✅ Action {i+1} completed successfully")
                    logger.info(f"| 📄 Results: {str(tool_result)[:200]}...")
                    
                    # Update action with result
                    action_dict = action.model_dump()
                    action_dict["output"] = tool_result
                    action_results.append(action_dict)
                    
                except Exception as e:
                    logger.error(f"| ❌ Error executing action {action.name}: {e}")
                    action_results.append({
                        "name": action.name,
                        "error": str(e),
                        "output": None
                    })
                    
                    # Ask user how to handle the error
                    if self.interactive_mode:
                        await self._handle_error_interaction(e, action.name)
            
            # Record event in memory
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
            
            if done:
                self.memory_manager.add_event(
                    step_number=self.step_number,
                    event_type="task_end",
                    data=dict(result=final_result),
                    agent_name=self.name,
                    task_id=task_id
                )
            
        except Exception as e:
            logger.error(f"| ❌ Error in thinking and action step: {e}")
            if self.interactive_mode:
                await self._handle_error_interaction(e, "thinking")
        
        return done, final_result
    
    async def _display_status(self, thinking: str, next_goal: str, actions: List):
        """Display current status in a user-friendly format."""
        logger.info("| 🎯 CURRENT STATUS")
        logger.info(f"| 💭 Thinking: {thinking[:150]}...")
        logger.info(f"| 🎯 Next Goal: {next_goal}")
        logger.info(f"| 🔧 Actions to execute: {len(actions)}")
        
        for i, action in enumerate(actions):
            logger.info(f"| {i+1}. {action.name}")
    
    async def _handle_error_interaction(self, error: Exception, context: str):
        """Handle errors with user interaction."""
        logger.error(f"| ❌ Error in {context}: {error}")
        
        if self.interactive_mode:
            logger.info("| 💡 How would you like to proceed?")
            logger.info("|  1. Retry the action")
            logger.info("|  2. Skip and continue")
            logger.info("|  3. Modify the task")
            logger.info("|  4. Quit")
            
            # In a real implementation, you'd get user input here
            # For now, we'll just log the options
            logger.info("|   (Interactive input would be handled here)")
    
    async def _get_user_input(self) -> str:
        """Get user input for interactive control."""
        # This would be implemented based on your UI framework
        # For now, return a default action
        return "continue"
    
    async def _ask_user_continue(self, iteration: int) -> bool:
        """Ask user whether to continue to next iteration."""
        if not self.interactive_mode or self.auto_continue:
            return True
        
        logger.info(f"| 🔄 Iteration {iteration} completed.")
        logger.info("| Continue to next iteration? (y/n/q for quit)")
        
        # In real implementation, get user input
        # For now, return True to continue
        return True
    
    async def run(self, task: str):
        """Run the interactive agent with user interaction."""
        logger.info(f"| 🚀 Starting InteractiveAgent: {task}")
        logger.info(f"| 🎮 Interactive mode: {'ON' if self.interactive_mode else 'OFF'}")
        
        session_info = self._generate_session_info(task)
        session_id = session_info.session_id
        description = session_info.description
        
        # Start session
        self.memory_manager.start_session(session_id, description)
        
        # Add task start event
        task_id = str(uuid.uuid4())
        self.memory_manager.add_event(
            step_number=self.step_number, 
            event_type="task_start", 
            data=dict(task=task),
            agent_name=self.name,
            task_id=task_id
        )
        
        # Initialize messages
        messages = self._get_messages(task)
        
        # Main interactive loop
        iteration = 0
        done = False
        final_result = None
        
        while iteration < self.max_iterations and not done:
            iteration += 1
            logger.info(f"| 🔄 Iteration {iteration}/{self.max_iterations}")
            
            # Execute one step
            done, final_result = await self._think_and_action(messages, task_id)
            self.step_number += 1
            
            # Update messages for next iteration
            messages = self._get_messages(task)
            
            # Ask user whether to continue (if interactive mode)
            if not done and iteration < self.max_iterations:
                should_continue = await self._ask_user_continue(iteration)
                if not should_continue:
                    logger.info("| 👋 User chose to stop. Ending session.")
                    break
        
        # Handle max iterations reached
        if iteration >= self.max_iterations:
            logger.warning(f"| 🛑 Reached max iterations ({self.max_iterations}), stopping...")
            final_result = "Reached maximum number of iterations"
        
        # Add final task end event
        self.memory_manager.add_event(
            step_number=self.step_number,
            event_type="task_end",
            data=dict(result=final_result),
            agent_name=self.name,
            task_id=task_id
        )
        
        # End session
        self.memory_manager.end_session()
        
        logger.info(f"| ✅ Interactive agent completed after {iteration}/{self.max_iterations} iterations")
        
        return final_result
    
    async def run_streaming(self, task: str):
        """Run the agent with streaming output (for real-time display)."""
        # This would implement streaming output similar to Cursor
        # For now, just call the regular run method
        return await self.run(task)
