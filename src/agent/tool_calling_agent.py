"""Tool calling agent implementation with manual agent logic."""

import asyncio
import os
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import Field, ConfigDict

from src.agent.types import Agent, AgentResponse, AgentExtra, ThinkOutput
from src.config import config
from src.logger import logger
from src.utils import dedent, parse_tool_args
from src.tool.server import tcp
from src.environment.server import ecp
from src.memory import memory_manager, EventType
from src.tracer import Tracer, Record
from src.model import model_manager
from src.registry import AGENT
from src.session import SessionContext

@AGENT.register_module(force=True)
class ToolCallingAgent(Agent):
    """Tool calling agent implementation with manual agent logic."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="tool_calling", description="The name of the tool calling agent.")
    description: str = Field(default="A tool calling agent that can call tools to complete tasks.", description="The description of the tool calling agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool calling agent.")
    require_grad: bool = Field(default=False, description="Whether the agent requires gradients")
    
    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        max_tools: int = 10,
        max_steps: int = 20,
        review_steps: int = 5,
        require_grad: bool = False,
        **kwargs
    ):
        # Set default prompt name for tool calling
        if not prompt_name:
            prompt_name = "tool_calling"
        
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_name=memory_name,
            max_tools=max_tools,
            max_steps=max_steps,
            review_steps=review_steps,
            require_grad=require_grad,
            **kwargs)
    
    async def initialize(self):
        """Initialize the agent."""
        self.tracer_save_path = os.path.join(self.workdir, "tracer.json")
        await super().initialize()
    
    async def _get_tracer_and_record(self) -> tuple[Tracer, Record]:
        """Get tracer and record for current call (coroutine-safe)."""
        tracer = Tracer()
        record = Record()
        
        if os.path.exists(self.tracer_save_path):
            await tracer.load_from_json(self.tracer_save_path)
            last_record = await tracer.get_last_record()
            if last_record:
                record = last_record
        
        return tracer, record
    
    async def _get_environment_context(self, ctx: SessionContext, record: Record = None, **kwargs) -> Dict[str, Any]:
        """Get the environment state."""
        
        environment_context = "<environment_context>"
        record_observation = {}
        
        # Only iterate over environments specified in config, not all registered environments
        for env_name in config.env_names:
            env_info = await ecp.get_info(env_name)
            rule_string = env_info.rules
            rule_string = dedent(f"""
                <rules>
                {rule_string}
                </rules>
            """)
            
            env_state = await ecp.get_state(env_name, ctx=ctx)
            state_string = "<state>"
            state_string += env_state["state"]
            extra = env_state["extra"]
            record_observation[env_name] = extra
            
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
        
        if record is not None:
            record.observation = record_observation
        
        environment_context += "</environment_context>"
        return {
            "environment_context": environment_context,
        }
        
    async def _get_tool_context(self, ctx: SessionContext, record: Record = None, **kwargs) -> Dict[str, Any]:
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
        
    async def _think_and_tool(self, 
                              messages: List[BaseMessage], 
                              task_id: str,
                              step_number: int,
                              record: Record = None, 
                              ctx: SessionContext = None, 
                              **kwargs)->Dict[str, Any]:
        """Think and tool calls for one step."""
        
        done = False
        result = None
        reasoning = None
        
        record_tool = {
            "thinking": None,
            "evaluation_previous_goal": None,
            "memory": None,
            "next_goal": None,
            "tool": [],
        }
        
        try:
            think_output = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=ThinkOutput
            )
            think_output = think_output.extra.parsed_model
            
            thinking = think_output.thinking
            evaluation_previous_goal = think_output.evaluation_previous_goal
            memory = think_output.memory
            next_goal = think_output.next_goal
            tools = think_output.tool
            
            # Update record tool
            record_tool["thinking"] = thinking
            record_tool["evaluation_previous_goal"] = evaluation_previous_goal
            record_tool["memory"] = memory
            record_tool["next_goal"] = next_goal
            
            logger.info(f"| 💭 Thinking: {thinking}")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 🔧 Tools to execute: {tools}")
            
            # Execute tools sequentially
            tool_results = []
            
            for i, tool in enumerate(tools):
                logger.info(f"| 📝 Tool {i+1}/{len(tools)}: {tool.name}")
                
                # Execute the tool
                tool_name = tool.name
                tool_args_str = tool.args
                if tool_args_str:
                    tool_args = parse_tool_args(tool_args_str)
                else:
                    tool_args = {}
                
                logger.info(f"| 📝 Tool Name: {tool_name}, Args: {tool_args}")
                
                input = {
                    "name": tool_name,
                    "input": tool_args,
                    "ctx": ctx
                }
                tool_response = await tcp(**input)
                tool_result = tool_response.message
                tool_extra = tool_response.extra if hasattr(tool_response, 'extra') else None
                
                logger.info(f"| ✅ Tool {i+1} completed successfully")
                logger.info(f"| 📄 Results: {str(tool_result)}")
                
                # Update tool with result
                tool_dict = tool.model_dump()
                tool_dict["output"] = tool_result
                tool_results.append(tool_dict)
                
                # Update record tool
                tool_extra_dict = {}
                tool_extra_dict.update(tool_dict)
                if tool_extra is not None:
                    tool_extra_dict['extra'] = tool_extra.model_dump()
                record_tool["tool"].append(tool_extra_dict)
                    
                if tool_name == "done":
                    done = True
                    result = tool_result
                    reasoning = tool_extra.data.get('reasoning', None) if tool_extra and tool_extra.data else None
                    break
            
            event_data = {
                "thinking": thinking,
                "evaluation_previous_goal": evaluation_previous_goal,
                "memory": memory,
                "next_goal": next_goal,
                "tool": tool_results
            }
            
            # Update record tool
            if record is not None:
                record.tool = record_tool
            
            # Get memory system name
            memory_name = self.memory_name
            
            # Add event to memory if use_memory is enabled
            if self.use_memory and memory_name:
                await memory_manager.add_event(
                    memory_name=memory_name,
                    step_number=step_number,
                    event_type=EventType.TOOL_STEP,
                    data=event_data,
                    agent_name=self.name,
                    task_id=task_id,
                    ctx=ctx
                )
            
        except Exception as e:
            logger.error(f"| Error in thinking and tool step: {e}")
        
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
        Main entry point for tool calling agent through acp.
        
        Args:
            task (str): The task to complete.
            files (Optional[List[str]]): The files to attach to the task.
            
        Returns:
            AgentResponse: The response of the agent.
        """
        logger.info(f"| 🚀 Starting ToolCallingAgent: {task}")
        
        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()
        
        # Create tracer and record as local variables (coroutine-safe)
        tracer, record = await self._get_tracer_and_record()
        
        if files:
            logger.info(f"| 📂 Attached files: {files}")
            files = await asyncio.gather(*[self._extract_file_content(file) for file in files])
            enhanced_task = await self._generate_enhanced_task(task, files)
        else:
            enhanced_task = task
        
        # Get memory system name
        memory_name = self.memory_name

        task_id = "task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        logger.info(f"| 📝 Context ID: {ctx.id}, Task ID: {task_id}")
        
        # Memory session management (only if use_memory is enabled)
        if self.use_memory and memory_name:
            await memory_manager.start_session(memory_name=memory_name, ctx=ctx)
            
            # Add task start event
            await memory_manager.add_event(
                memory_name=memory_name,
                step_number=0,
                event_type=EventType.TASK_START,
                data=dict(task=enhanced_task),
                agent_name=self.name,
                task_id=task_id,
                ctx=ctx
            )
        else:
            logger.info(f"| ⏭️ Memory disabled (use_memory={self.use_memory}), skipping session management")
        
        # Initialize messages
        messages = await self._get_messages(enhanced_task, ctx=ctx)
        
        # Main loop
        step_number = 0
        
        while step_number < self.max_steps:
            logger.info(f"| 🔄 Step {step_number+1}/{self.max_steps}")
            
            # Execute one step
            response = await self._think_and_tool(messages, task_id, step_number, ctx=ctx, record=record)
            step_number += 1
            
            # Update tracer and save to json
            await tracer.add_record(observation=record.observation, 
                                        tool=record.tool,
                                        task_id=task_id,
                                        ctx=ctx)
            await tracer.save_to_json(self.tracer_save_path)
            
            # Memory is automatically saved in add_event()
            messages = await self._get_messages(enhanced_task, ctx=ctx)
            
            if response["done"]:
                break
        
        # Handle max steps reached
        if step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping...")
            response = {
                "done": False,
                "result": "The task has not been completed.",
                "reasoning": "Reached the maximum number of steps."
            }
        
        # Get memory system name
        memory_name = self.memory_name
        
        # Add task end event and end session (only if use_memory is enabled)
        if self.use_memory and memory_name:
            await memory_manager.add_event(
                memory_name=memory_name,
                step_number=step_number,
                event_type=EventType.TASK_END,
                data=response,
                agent_name=self.name,
                task_id=task_id,
                ctx=ctx
            )
            
            # End session (automatically saves memory to JSON)
            await memory_manager.end_session(memory_name=memory_name, ctx=ctx)
        
        # Save tracer to json
        await tracer.save_to_json(self.tracer_save_path)
        
        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")
        
        return AgentResponse(
            success=response["done"],
            message=response["result"],
            extra=AgentExtra(
                data=response
            )
        )