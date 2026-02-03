"""Planning agent implementation for task decomposition and sub-agent coordination."""

import asyncio
import os
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import Field, ConfigDict

from src.agent.types import Agent, AgentResponse, AgentExtra, ThinkOutput
from src.config import config
from src.logger import logger
from src.utils import dedent
from src.tool.server import tcp
from src.agent.server import acp
from src.environment.server import ecp
from src.memory import memory_manager, EventType
from src.tool.types import ToolResponse
from src.tracer import Tracer, Record
from src.model import model_manager
from src.registry import AGENT
from src.session import SessionContext

@AGENT.register_module(force=True)
class PlanningAgent(Agent):
    """Planning agent for high-level reasoning, task decomposition, and adaptive planning."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="planning", description="The name of the planning agent.")
    description: str = Field(default="A planning agent that decomposes complex tasks and coordinates sub-agents.", description="The description of the planning agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the planning agent.")
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
        # Set default prompt name for planning
        if not prompt_name:
            prompt_name = "planning"
        
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
    
    async def _get_agent_context(self, task: str, ctx: SessionContext = None, **kwargs) -> Dict[str, Any]:
        """Get the agent context including available agents."""
        # Get base agent context from parent
        base_context = await super()._get_agent_context(task, ctx=ctx)
        
        # Extract the base agent context string
        base_agent_context = base_context["agent_context"]
        
        # Add available agents information
        available_agents_info = ""
        try:
            available_agents = await acp.list()
            agent_contract = await acp.get_contract()
            
            available_agents_info = dedent(f"""
                <available_agents>
                Available sub-agents that can be called to complete subtasks:
                
                {agent_contract}
                
                To call a sub-agent, use the agent's name as a tool. The agent will be invoked through ACP.
                Each agent accepts a 'task' parameter (required) and optional 'files' parameter.
                </available_agents>
            """)
        except Exception as e:
            logger.warning(f"| ⚠️ Could not get agent contract: {e}")
            available_agents_info = dedent(f"""
                <available_agents>
                Agent information is currently unavailable. You can still use tools and the todo tool for planning.
                </available_agents>
            """)
        
        # Insert available agents info before the closing </agent_context> tag
        # Find the last occurrence of </agent_context> and insert before it
        agent_context = base_agent_context.replace("</agent_context>", available_agents_info + "\n</agent_context>")
        
        return {
            "agent_context": agent_context,
        }
    
    async def _get_environment_context(self, record: Record = None) -> Dict[str, Any]:
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
        
    async def _think_and_tool(self, messages: List[BaseMessage], task_id: str, step_number: int, ctx: SessionContext = None, record: Record = None) -> Dict[str, Any]:
        """Think and tool calls for one step, with support for agent calls."""
        
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
            
            logger.info(f"| 💭 Thinking: {thinking[:1000]}...")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 🔧 Tools/Agents to execute: {len(tools)}")
            
            # Execute tools/agents sequentially
            tool_results = []
            
            for i, tool in enumerate(tools):
                logger.info(f"| 📝 Tool/Agent {i+1}/{len(tools)}: {tool.name}")
                
                # Execute the tool or agent
                tool_name = tool.name
                tool_args = tool.args if tool.args else {}
                
                logger.info(f"| 📝 Tool/Agent Name: {tool_name}, Args: {tool_args}")
                
                # Check if this is an agent call (agents are available as tools through A2T transformation)
                # Or we can check if it's in the agent list
                try:
                    # Try to get agent info to see if this is an agent
                    agent_info = await acp.get_info(tool_name)
                    if agent_info:
                        # This is an agent call through ACP
                        logger.info(f"| 🤖 Calling agent: {tool_name}")
                        agent_task = tool_args.get("task", "")
                        agent_files = tool_args.get("files", None)
                        
                        agent_result = await acp(
                            name=tool_name,
                            input={
                                "task": agent_task,
                                "files": agent_files
                            }
                        )
                        
                        # Convert AgentResponse to tool-like result
                        if hasattr(agent_result, 'success'):
                            tool_result = agent_result.message if agent_result.success else f"Agent call failed: {agent_result.message}"
                        else:
                            tool_result = str(agent_result)
                        
                        logger.info(f"| ✅ Agent {i+1} completed successfully")
                        logger.info(f"| 📄 Results: {str(tool_result)[:1000]}...")
                        
                        # Update tool with result
                        tool_dict = tool.model_dump()
                        tool_dict["output"] = tool_result
                        tool_dict["agent_call"] = True
                        tool_results.append(tool_dict)
                        
                        # Update record tool
                        tool_extra_dict = {}
                        tool_extra_dict.update(tool_dict)
                        tool_extra_dict["agent_call"] = True
                        record_tool["tool"].append(tool_extra_dict)
                    else:
                        # This is a regular tool call
                        input = {
                            "name": tool_name,
                            "input": tool_args,
                            "ctx": ctx
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
                except Exception as e:
                    # If agent lookup fails, treat as regular tool
                    logger.info(f"| 🔧 Treating {tool_name} as regular tool (agent lookup failed: {e})")
                    input = {
                        "name": tool_name,
                        "input": tool_args,
                        "ctx": ctx
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
            
            # Get memory system name
            memory_name = self.memory_name
            
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
        """Main entry point for planning agent through acp."""
        logger.info(f"| 🚀 Starting PlanningAgent: {task}")
        
        # Create tracer and record as local variables (coroutine-safe)
        tracer, record = await self._get_tracer_and_record()
        
        if files:
            logger.info(f"| 📂 Attached files: {files}")
            files = await asyncio.gather(*[self._extract_file_content(file) for file in files])
            enhanced_task = await self._generate_enhanced_task(task, files)
        else:
            enhanced_task = task
        
        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()
        task_id = "task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        logger.info(f"| 📝 Context ID: {ctx.id}, Task ID: {task_id}")
        
        # Get memory system name
        memory_name = self.memory_name
        
        # Start session
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
        
        # Initialize messages
        messages = await self._get_messages(enhanced_task, ctx=ctx, record=record)
        
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
            
            messages = await self._get_messages(enhanced_task, ctx=ctx, record=record)
            
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
        
        # Add task end event
        await memory_manager.add_event(
            memory_name=memory_name,
            step_number=step_number,
            event_type=EventType.TASK_END,
            data=response,
            agent_name=self.name,
            task_id=task_id,
            ctx=ctx
        )
        
        # End session
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

