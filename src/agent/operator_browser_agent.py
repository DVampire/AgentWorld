"""Operator Browser Agent implementation for web automation tasks using vision-enabled LLM."""

import asyncio
from typing import List, Optional, Type, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from src.agent.types import Agent, InputArgs, AgentResponse, AgentExtra, AgentContext
from src.logger import logger
from src.utils import get_file_info, dedent
from src.agent.server import acp
from src.tool.server import tcp
from src.environment.server import ecp
from src.memory import memory_manager, EventType
from src.memory.types import MemoryContext
from src.tool.types import ToolResponse, ToolContext
from src.prompt import prompt_manager
from src.model import model_manager
from src.registry import AGENT

@AGENT.register_module(force=True)
class OperatorBrowserAgent(Agent):
    """Operator Browser Agent implementation with visual understanding capabilities."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="operator_browser", description="The name of the operator browser agent.")
    description: str = Field(default="A browser operator agent that can see and control web browsers using vision-enabled LLM.", description="The description of the operator browser agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the operator browser agent.")
    require_grad: bool = Field(default=False, description="Whether the agent requires gradients")
    
    def __init__(
        self,
        workdir: str,
        model_name: Optional[str] = "computer-browser-use",
        prompt_name: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        max_steps: int = 30,
        review_steps: int = 5,
        if_correct_action: bool = True,
        require_grad: bool = False,
        **kwargs
    ):
        """Initialize the Operator Browser Agent.
        
        Args:
            workdir: Working directory for logs and screenshots
            model_name: LLM model name (should support vision, default: computer-browser-use)
            prompt_name: Name of the prompt template (default: operator_browser)
            memory_config: Memory configuration
            max_steps: Maximum number of steps
            review_steps: Number of steps to review in history
            if_correct_action: If True, correct the action
        """
        # Set default prompt name for operator browser
        if not prompt_name:
            prompt_name = "operator_browser"
        
        super().__init__(
            workdir=workdir,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_config=memory_config,
            max_steps=max_steps,
            review_steps=review_steps,
            require_grad=require_grad,
            **kwargs)
        
        self.if_correct_action = if_correct_action
    
    async def _extract_file_content(self, file: str) -> str:
        """Extract file information."""
        
        info = get_file_info(file)
        
        # Extract file content
        input = {
            "name": "mdify",
            "input": {
                "file_path": file,
                "output_format": "markdown"
            }
        }
        tool_response = await tcp(**input)
        file_content = tool_response.message
        
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
        
        model_response = await model_manager(model=self.model_name, messages=messages)
        
        info["content"] = file_content
        info["summary"] = model_response.message
        
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
    
    async def _get_agent_history(self, ctx: MemoryContext = None) -> Dict[str, Any]:
        """Get the agent history."""
        state = await memory_manager.get_state(memory_name=self.memory_name, n=self.review_steps, ctx=ctx)
        
        events = state["events"]
        summaries = state["summaries"]
        insights = state["insights"]
        
        agent_history = ""
        for event in events:
            agent_history += f"<step_{event.step_number}>\n"
            if event.event_type == EventType.TASK_START:
                agent_history += f"Task Start: {event.data.get('task', event.data.get('message', ''))}\n"
            elif event.event_type == EventType.TASK_END:
                agent_history += f"Task End: {event.data.get('result', '')}\n"
            elif event.event_type == EventType.ACTION_STEP:
                agent_history += f"Action Results: {event.data.get('action', '')}\n"
                agent_history += f"Reasoning: {event.data.get('reasoning', '')}\n"
            agent_history += "\n"
            agent_history += f"</step_{event.step_number}>\n"
        
        if len(summaries) > 0:
            agent_history += dedent(f"""
                <summaries>
                {chr(10).join([str(summary) for summary in summaries])}
                </summaries>
            """)
        if len(insights) > 0:
            agent_history += dedent(f"""
                <insights>
                {chr(10).join([str(insight) for insight in insights])}
                </insights>
            """)
        
        return {
            "agent_history": agent_history,
        }
    
    async def _get_todo_contents(self, id: Optional[str] = None) -> str:
        """Get the todo contents for a specific id."""
        todo_tool = await tcp.get("todo")
        if id:
            todo_contents = todo_tool.get_todo_content(id)
        else:
            todo_contents = "[Current todo.md is empty, fill it with your plan when applicable]"
        return todo_contents   
    
    async def _get_agent_state(self, task: str, step_number: Optional[int] = None) -> Dict[str, Any]:
        """Get the agent state."""
        current_step = step_number if step_number is not None else self.step_number
        step_info_description = f'Step {current_step + 1} of {self.max_steps} max possible steps\n'
        time_str = datetime.now().isoformat()
        step_info_description += f'Current date and time: {time_str}'
        
        available_actions_description = [tcp.to_string(tool) for tool in tcp.list()]
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
        
        for env_name in ecp.list():
            
            env_state = await ecp.get_state(env_name)
            state_string = env_state["state"]
            extra = env_state["extra"]
            
            # CUA only supports the one latest screenshot
            if "screenshots" in extra:
                last_screenshot = extra["screenshots"][-1]
                state_string += f"\n<img src={last_screenshot.screenshot_path} alt={last_screenshot.screenshot_description}/>"
                        
            environment_state += dedent(f"""
                <{env_name}_state>
                {state_string}
                </{env_name}_state>
            """)
        
        return {
            "environment_state": environment_state,
        }
        
    async def _get_messages(self, task: str, ctx: AgentContext = None, **kwargs) -> List[BaseMessage]:
        
        id = ctx.id if ctx else None
        step_number = ctx.step_number if ctx else None
        memory_ctx = MemoryContext(id=id) if id else None
        
        system_modules = {}
        # Infer prompt name from agent's prompt_name
        if self.prompt_name:
            system_prompt_name = f"{self.prompt_name}_system_prompt"
            agent_message_prompt_name = f"{self.prompt_name}_agent_message_prompt"
        else:
            system_prompt_name = "operator_browser_system_prompt"
            agent_message_prompt_name = "operator_browser_agent_message_prompt"
        
        # Add environment rules to system modules
        environment_rules = ""
        for env_name in ecp.list():
            env_info = await ecp.get_info(env_name)
            environment_rules += f"{env_info.rules}\n"
        system_modules.update(dict(
            environment_rules=environment_rules,
        ))
        
        system_message = await prompt_manager.get_system_message(
            prompt_name=system_prompt_name,
            modules=system_modules, 
            reload=False
        )
        
        agent_message_modules = {}
        agent_history = await self._get_agent_history(ctx=memory_ctx)
        agent_state = await self._get_agent_state(task, step_number=step_number)
        environment_state = await self._get_environment_state()
        
        agent_message_modules.update(agent_history)
        agent_message_modules.update(agent_state)
        agent_message_modules.update(environment_state)
        
        agent_message = await prompt_manager.get_agent_message(
            prompt_name=agent_message_prompt_name,
            modules=agent_message_modules, 
            reload=True
        )
        
        messages = [
            system_message,
            agent_message,
        ]
        
        return messages
    
    async def _think_and_action(self, messages: List[BaseMessage], task_id: str, ctx: AgentContext = None) -> Dict[str, Any]:
        """Think and action for one step."""
        
        id = ctx.id if ctx else None
        step_number = ctx.step_number if ctx else None
        memory_ctx = MemoryContext(id=id) if id else None
        tool_ctx = ToolContext(id=id) if id else None
        
        done = False
        result = None
        reasoning = None
        
        current_step = step_number if step_number is not None else self.step_number
        
        try:
            model_response = await model_manager(model=self.model_name, messages=messages)
            response = model_response.message
            
            reasoning = ""
            action = {}
            # Handle both string and dict responses
            if isinstance(response, str):
                contents = response
            elif hasattr(response, 'content'):
                contents = response.content
            else:
                contents = str(response)
            
            if isinstance(contents, list):
                for content in contents:
                    if content['type'] == 'text':
                        reasoning += content["text"]
                    elif content['type'] == 'reasoning':
                        summary = " ".join([item['text'] for item in content["summary"]])
                        reasoning += summary
                    elif content['type'] == 'computer_call':
                        action_dict = content['action']
                        action['name'] = action_dict['type']
                        action['args'] = {
                            key: value for key, value in action_dict.items() if key != 'type'
                        }
            
            elif isinstance(contents, str):
                reasoning += contents
                action['name'] = "wait"
                action['args'] = {"ms": 1000} # wait for 1 second
            
            logger.info(f"| 💭 Reasoning: {reasoning}")
            logger.info(f"| 🎯 Action: {action}")
            
            # Get tool name and args
            tool_name = action['name']
            tool_args = action['args']
                
            # Execute the first action
            action_results = []
            
            # Auto-inject id for todo tool using ctx.id
            if tool_name == "todo" and id:
                tool_args["id"] = id
            
            logger.info(f"| 📝 Action Name: {tool_name}, Args: {tool_args}")
            
            input = {
                "name": tool_name,
                "input": tool_args,
                "ctx": tool_ctx
            }
            tool_response = await tcp(**input)
            tool_result = tool_response.message
            tool_extra = tool_response.extra if hasattr(tool_response, 'extra') else None
            
            logger.info(f"| ✅ Action completed successfully")
            logger.info(f"| 📄 Results: {str(tool_result)}")
            
            # Update action with result
            action["output"] = tool_result
            action_results.append(action)
                
            if tool_name == "done":
                done = True
                result = tool_result
                reasoning = tool_extra.data.get('reasoning', None) if tool_extra and tool_extra.data else None
            
            event_data = {
                "reasoning": reasoning,
                "action": action_results
            }
            await memory_manager.add_event(
                memory_name=self.memory_name,
                step_number=current_step,
                event_type=EventType.TOOL_STEP,
                data=event_data,
                agent_name=self.name,
                task_id=task_id,
                ctx=memory_ctx
            )
            
            if done:
                await memory_manager.add_event(
                    memory_name=self.memory_name,
                    step_number=current_step + 1,
                    event_type=EventType.TASK_END,
                    data=dict(result=result),
                    agent_name=self.name,
                    task_id=task_id,
                    ctx=memory_ctx
                )
            
        except Exception as e:
            logger.error(f"| Error in thinking and action step: {e}")
        
        response_dict = {
            "done": done,
            "result": result,
            "reasoning": reasoning
        }
        return response_dict
    
    async def __call__(self, 
                  task: str, 
                  files: Optional[List[str]] = None,
                  **kwargs
                  ) -> AgentResponse:
        """
        Main entry point for operator browser agent through acp.
        
        Args:
            task (str): The task to complete.
            files (Optional[List[str]]): The files to attach to the task.
            ctx (AgentContext): The agent context.
            
        Returns:
            AgentResponse: The response of the agent.
        """
        logger.info(f"| 🚀 Starting OperatorBrowserAgent: {task}")
        
        if files:
            logger.info(f"| 📂 Attached files: {files}")
            files = await asyncio.gather(*[self._extract_file_content(file) for file in files])
            enhanced_task = await self._generate_enhanced_task(task, files)
        else:
            enhanced_task = task
        
        # Get id from ctx
        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = AgentContext()
        id = ctx.id
        task_id = "task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        memory_ctx = MemoryContext(id=id)
        
        logger.info(f"| 📝 Context ID: {id}, Task ID: {task_id}")
        
        # Start session
        await memory_manager.start_session(memory_name=self.memory_name, ctx=memory_ctx)
        
        # Add task start event
        await memory_manager.add_event(
            memory_name=self.memory_name,
            step_number=0, 
            event_type=EventType.TASK_START, 
            data=dict(task=enhanced_task),
            agent_name=self.name,
            task_id=task_id,
            ctx=memory_ctx
        )
        
        # Initialize messages
        ctx.step_number = 0
        messages = await self._get_messages(enhanced_task, ctx=ctx)

        # Main loop
        step_number = 0
        response = None
        
        while step_number < self.max_steps:
            logger.info(f"| 🔄 Step {step_number+1}/{self.max_steps}")
            
            # Execute one step
            response = await self._think_and_action(messages, task_id, ctx=ctx)
            step_number += 1
            
            # Update ctx step_number
            ctx.step_number = step_number
            messages = await self._get_messages(enhanced_task, ctx=ctx)
            
            if response["done"]:
                break
        
        # Handle max steps reached
        if step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping...")
            response = {
                "done": False,
                "result": "Reached maximum number of steps",
                "reasoning": "Reached the maximum number of steps."
            }
        
        # Add task end event
        await memory_manager.add_event(
            memory_name=self.memory_name,
            step_number=step_number,
            event_type=EventType.TASK_END,
            data=response,
            agent_name=self.name,
            task_id=task_id,
            ctx=memory_ctx
        )
        
        # End session
        await memory_manager.end_session(memory_name=self.memory_name, ctx=memory_ctx)
        
        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")
        
        return AgentResponse(
            success=response["done"],
            message=response["result"] if response["result"] else "Task completed",
            extra=AgentExtra(data=response)
        )