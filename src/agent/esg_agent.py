"""ESG Agent implementation - Specialized agent for ESG data analysis and report generation."""

import asyncio
import os
import json
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import Field, ConfigDict

from src.agent.types import Agent, AgentResponse, AgentExtra, ThinkOutput, AgentContext
from src.config import config
from src.logger import logger
from src.utils import dedent
from src.tool.server import tcp
from src.environment.server import ecp
from src.memory import memory_manager, EventType
from src.memory.types import MemoryContext
from src.tool.types import ToolResponse, ToolContext
from src.environment.types import EnvironmentContext
from src.tracer import Tracer, Record
from src.model import model_manager
from src.registry import AGENT


@AGENT.register_module(force=True)
class ESGAgent(Agent):
    """ESG Agent implementation - specialized for ESG data analysis and report generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="esg_agent", description="The name of the ESG agent.")
    description: str = Field(
        default="An ESG agent specialized in retrieving, analyzing, and generating reports from ESG data.", 
        description="The description of the ESG agent."
    )
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the ESG agent.")
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
        max_steps: int = 30,
        review_steps: int = 5,
        require_grad: bool = False,
        **kwargs
    ):
        """Initialize the ESG Agent.
        
        Args:
            workdir: Working directory for the agent.
            name: Agent name.
            description: Agent description.
            metadata: Additional metadata.
            model_name: LLM model to use.
            prompt_name: Prompt template name (defaults to 'esg_agent').
            memory_name: Memory system name.
            max_tools: Maximum tools per step.
            max_steps: Maximum reasoning steps.
            review_steps: Steps between reviews.
        """
        # Set default prompt name for ESG agent
        if not prompt_name:
            prompt_name = "esg_agent"
        
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
            **kwargs
        )
        
        self.tracer_save_path = os.path.join(self.workdir, "tracer.json")
    
    async def initialize(self):
        """Initialize the ESG agent."""
        await super().initialize()
        logger.info(f"| 🌱 ESG Agent initialized: {self.name}")
    
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
    
    async def _get_environment_context(self, ctx: AgentContext = None, record: Record = None) -> Dict[str, Any]:
        """Get the environment state for ESG analysis."""
        
        id = ctx.id if ctx else None
        environment_ctx = EnvironmentContext(id=id) if id else None
        
        environment_context = "<environment_context>"
        
        record_observation = {}
        
        # Only iterate over environments specified in config
        for env_name in config.env_names:
            env_info = await ecp.get_info(env_name)
            rule_string = env_info.rules
            rule_string = dedent(f"""
                <rules>
                {rule_string}
                </rules>
            """)
            
            env_state = await ecp.get_state(env_name, ctx=environment_ctx)
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
        
    async def _think_and_tool(self, messages: List[BaseMessage], task_id: str, ctx: AgentContext = None, record: Record = None) -> Dict[str, Any]:
        """Execute one ESG analysis step - think and call tools."""
        
        id = ctx.id if ctx else None
        step_number = ctx.step_number if ctx else None
        memory_ctx = MemoryContext(id=id) if id else None
        tool_ctx = ToolContext(id=id) if id else None
        
        done = False
        result = None
        reasoning = None
        
        # Use provided step_number or fallback to instance state
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
            
            logger.info(f"| 💭 ESG Thinking: {thinking[:1000]}...")
            logger.info(f"| 🎯 Next ESG Goal: {next_goal}")
            logger.info(f"| 🔧 ESG Tools to execute: {len(tools)}")
            
            # Execute tools sequentially
            tool_results = []
            
            for i, tool in enumerate(tools):
                logger.info(f"| 📝 ESG Tool {i+1}/{len(tools)}: {tool.name}")
                
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
                
                logger.info(f"| ✅ ESG Tool {i+1} completed")
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
                step_number=current_step,
                event_type=EventType.TOOL_STEP,
                data=event_data,
                agent_name=self.name,
                task_id=task_id,
                ctx=memory_ctx
            )
            
        except Exception as e:
            logger.error(f"| ❌ Error in ESG analysis step: {e}")
        
        response_dict = {
            "done": done,
            "result": result,
            "reasoning": reasoning
        }
        return response_dict
        
    async def __call__(
        self, 
        task: str, 
        files: Optional[List[str]] = None,
        ctx: AgentContext = None,
        **kwargs
    ) -> AgentResponse:
        """
        Main entry point for ESG Agent.
        
        Args:
            task (str): The ESG analysis task to complete.
            files (Optional[List[str]]): Optional files to attach (e.g., ESG reports).
            ctx (AgentContext): The agent context.
            
        Returns:
            AgentResponse: The response of the agent.
        """
        logger.info(f"| 🌱 Starting ESG Agent: {task}")
        
        # Create tracer and record as local variables (coroutine-safe)
        tracer, record = await self._get_tracer_and_record()
        
        if files:
            logger.info(f"| 📂 Attached ESG files: {files}")
            files = await asyncio.gather(*[self._extract_file_content(file) for file in files])
            enhanced_task = await self._generate_enhanced_task(task, files)
        else:
            enhanced_task = task
        
        # Get memory system name
        memory_name = self.memory_name
        
        # Get id from ctx
        if ctx is None:
            ctx = AgentContext()
        id = ctx.id
        task_id = "esg_task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        memory_ctx = MemoryContext(id=id)
        
        logger.info(f"| 📝 Context ID: {id}, Task ID: {task_id}")
        
        # Start session
        await memory_manager.start_session(memory_name=memory_name, ctx=memory_ctx)
        
        # Add task start event
        await memory_manager.add_event(
            memory_name=memory_name,
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
            logger.info(f"| 🔄 ESG Analysis Step {step_number+1}/{self.max_steps}")
            
            # Execute one step
            response = await self._think_and_tool(messages, task_id, ctx=ctx, record=record)
            step_number += 1
            
            # Update tracer and save to json
            await tracer.add_record(
                observation=record.observation, 
                tool=record.tool,
                session_id=ctx.id,
                task_id=task_id
            )
            await tracer.save_to_json(self.tracer_save_path)
            
            # Update ctx step_number
            ctx.step_number = step_number
            messages = await self._get_messages(enhanced_task, ctx=ctx)
            
            if response["done"]:
                break
        
        # Handle max steps reached
        if step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max ESG analysis steps ({self.max_steps}), stopping...")
            response = {
                "done": False,
                "result": "Reached maximum number of ESG analysis steps",
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
            ctx=memory_ctx
        )
        
        # End session
        await memory_manager.end_session(memory_name=memory_name, ctx=memory_ctx)
        
        # Save tracer to json
        await tracer.save_to_json(self.tracer_save_path)
        
        logger.info(f"| ✅ ESG Agent completed after {step_number}/{self.max_steps} steps")
        
        return AgentResponse(
            success=response["done"],
            message=response["result"] if response["result"] else "",
            extra=AgentExtra(data=response)
        )

