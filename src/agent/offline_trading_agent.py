"""Offline trading agent implementation for backtesting with historical data."""

import asyncio
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from pydantic import Field, ConfigDict

from src.logger import logger
from src.utils import dedent
from src.agent.server import acp
from src.tool.server import tcp
from src.environment.server import ecp
from src.agent.types import Agent, InputArgs, AgentResponse, AgentExtra, ThinkOutput, AgentContext
from src.tool.types import ToolResponse, ToolContext
from src.memory import memory_manager, EventType
from src.memory.types import MemoryContext
from src.tracer import Tracer, Record
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT

@AGENT.register_module(force=True)
class OfflineTradingAgent(Agent):
    """Offline trading agent implementation for backtesting with historical data.
    
    This agent is based on OnlineTradingAgent and uses the same implementation
    for backtesting with historical data.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="offline_trading", description="The name of the offline trading agent.")
    description: str = Field(default="An offline trading agent for backtesting with historical data.", description="The description of the offline trading agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the offline trading agent.")
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
            prompt_name = "online_trading"  # Use same prompt as online trading
        
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
        
        self.tracer_save_path = os.path.join(self.workdir, "tracer.json")
    
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
        
        
    async def _get_agent_context(self, task: str, ctx: AgentContext = None, **kwargs) -> Dict[str, Any]:
        """Get the agent context."""
        
        id = ctx.id if ctx else None
        step_number = ctx.step_number if ctx else None
        memory_ctx = MemoryContext(id=id) if id else None
        
        task = f"<task>{task}</task>"
        
        current_step = step_number if step_number is not None else self.step_number
        step_info_description = f'Step {current_step + 1} of {self.max_steps} max possible steps\n'
        time_str = datetime.now().isoformat()
        step_info_description += f'Current date and time: {time_str}'
        step_info = dedent(f"""
            <step_info>
            {step_info_description}
            </step_info>
        """)
        
        state = await memory_manager.get_state(memory_name=self.memory_name, n=self.review_steps, ctx=memory_ctx)
        
        events = state["events"]
        summaries = state["summaries"]
        insights = state["insights"]
        
        agent_history = "<agent_history>"
        for event in events:
            agent_history += f"<step_{event.step_number}>\n"
            if event.event_type == EventType.TASK_START:
                agent_history += f"Task Start: {event.data.get('task', event.data.get('message', ''))}\n"
            elif event.event_type == EventType.TASK_END:
                agent_history += f"Task End: {event.data.get('result', '')}\n"
            elif event.event_type == EventType.TOOL_STEP:
                agent_history += f"Thinking: {event.data.get('thinking', '')}\n"
                agent_history += f"Memory: {event.data.get('memory', '')}\n"
                agent_history += f"Tool: {event.data.get('tool', '')}\n"
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
        
        agent_context = dedent(f"""
            <agent_context>
            {task}
            {step_info}
            {agent_history}
            {memory}
            </agent_context>
        """)
        
        return {
            "agent_context": agent_context,
        }
        
    async def _get_environment_context(self, record: Record = None) -> Dict[str, Any]:
        """Get the environment state."""
        environment_context = "<environment_context>"
        
        record_observation = {}
        
        for env_name in await ecp.list():
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
        
    async def _get_messages(self, task: str, ctx: AgentContext = None, record: Record = None, **kwargs) -> List[BaseMessage]:
        
        system_modules = {}
        # Infer prompt name from agent's prompt_name
        if self.prompt_name:
            system_prompt_name = f"{self.prompt_name}_system_prompt"
            agent_message_prompt_name = f"{self.prompt_name}_agent_message_prompt"
        else:
            system_prompt_name = "offline_trading_system_prompt"
            agent_message_prompt_name = "offline_trading_agent_message_prompt"
        
        system_message = await prompt_manager.get_system_message(
            prompt_name=system_prompt_name,
            modules=system_modules, 
            reload=False
        )
        
        agent_message_modules = {}
        agent_message_modules.update(await self._get_agent_context(task, ctx=ctx))
        agent_message_modules.update(await self._get_environment_context(record=record))
        agent_message_modules.update(await self._get_tool_context())
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
    
    async def _think_and_action(self, messages: List[BaseMessage], task_id: str, ctx: AgentContext = None, record: Record = None) -> Dict[str, Any]:
        """Think and action for one step."""
        
        id = ctx.id if ctx else None
        step_number = ctx.step_number if ctx else None
        memory_ctx = MemoryContext(id=id) if id else None
        tool_ctx = ToolContext(id=id) if id else None
        
        done = False
        result = None
        reasoning = None
        
        current_step = step_number if step_number is not None else self.step_number
        
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
            
            logger.info(f"| 💭 Thinking: {thinking[:1000]}...")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 🔧 Tools to execute: {len(tools)}")
            
            # Execute tools sequentially
            tool_results = []
            
            for i, tool in enumerate(tools):
                logger.info(f"| 📝 Tool {i+1}/{len(tools)}: {tool.name}")
                
                # Execute the tool
                tool_name = tool.name
                tool_args = tool.args if tool.args else {}
                
                logger.info(f"| 📝 Tool Name: {tool_name}, Args: {tool_args}")
                
                input = {
                    "name": tool_name,
                    "input": tool_args,
                    "ctx": tool_ctx
                }
                tool_response = await tcp(**input)
                tool_result = tool_response.message
                tool_extra = tool_response.extra if hasattr(tool_response, 'extra') else None
                
                logger.info(f"| ✅ Tool {i+1} completed successfully")
                logger.info(f"| 📄 Results: {str(tool_result)[:1000]}...")
                
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
            
            await memory_manager.add_event(
                memory_name=self.memory_name,
                step_number=current_step,
                event_type=EventType.TOOL_STEP,
                data=event_data,
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
                  ctx: AgentContext = None,
                  **kwargs
                  ) -> AgentResponse:
        """
        Main entry point for offline trading agent through acp.
        """
        logger.info(f"| 🚀 Starting OfflineTradingAgent: {task}")
        
        # Create tracer and record as local variables (coroutine-safe)
        tracer, record = await self._get_tracer_and_record()
        
        if files:
            logger.info(f"| 📂 Attached files: {files}")
            files = await asyncio.gather(*[self._extract_file_content(file) for file in files])
            enhanced_task = await self._generate_enhanced_task(task, files)
        else:
            enhanced_task = task
        
        # Get id from ctx
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
        messages = await self._get_messages(enhanced_task, ctx=ctx, record=record)
        
        # Main loop
        step_number = 0
        response = None
        
        while step_number < self.max_steps:
            logger.info(f"| 🔄 Step {step_number+1}/{self.max_steps}")
            
            ctx.step_number = step_number
            
            # Execute one step
            response = await self._think_and_action(messages, task_id, ctx=ctx, record=record)
            step_number += 1
            
            # Update tracer and save to json
            await tracer.add_record(observation=record.observation, 
                                   tool=record.tool,
                                   session_id=id,
                                   task_id=task_id)
            await tracer.save_to_json(self.tracer_save_path)
            
            ctx.step_number = step_number
            messages = await self._get_messages(enhanced_task, ctx=ctx, record=record)
            
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
        
        # Save tracer to json
        await tracer.save_to_json(self.tracer_save_path)
        
        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")
        
        return AgentResponse(
            success=response["done"],
            message=response["result"] if response["result"] else "",
            extra=AgentExtra(data=response)
        )

