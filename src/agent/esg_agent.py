"""ESG Agent implementation - Specialized agent for ESG data analysis and report generation."""

import asyncio
import os
import json
from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import Field, ConfigDict

from src.agent.types import Agent
from src.config import config
from src.logger import logger
from src.utils import dedent
from src.tool.server import tcp
from src.environment.server import ecp
from src.memory import memory_manager
from src.tool.types import ToolResponse
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
    
    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_modules: Optional[Dict[str, Any]] = None,
        memory_name: Optional[str] = None,
        max_tools: int = 10,
        max_steps: int = 30,
        review_steps: int = 5,
        log_max_length: int = 1000,
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
            prompt_modules: Custom prompt modules.
            memory_name: Memory system name.
            max_tools: Maximum tools per step.
            max_steps: Maximum reasoning steps.
            review_steps: Steps between reviews.
            log_max_length: Maximum log message length.
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
            prompt_modules=prompt_modules,
            memory_name=memory_name,
            max_tools=max_tools,
            max_steps=max_steps,
            review_steps=review_steps,
            log_max_length=log_max_length,
            **kwargs
        )
        
        self.tracer_save_path = os.path.join(self.workdir, "tracer.json")
    
    async def initialize(self):
        """Initialize the ESG agent."""
        # Call parent initialize to setup think_output_builder
        await super().initialize()
        
        self.tracer = Tracer()
        self.record = Record()
        
        if os.path.exists(self.tracer_save_path):
            await self.tracer.load_from_json(self.tracer_save_path)
            # Get the last record from current session if any exist
            last_record = await self.tracer.get_last_record()
            if last_record:
                self.record = last_record
        
        logger.info(f"| 🌱 ESG Agent initialized: {self.name}")
    
    async def _get_environment_context(self) -> Dict[str, Any]:
        """Get the environment state for ESG analysis."""
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
        
        self.record.observation = record_observation
        
        environment_context += "</environment_context>"
        return {
            "environment_context": environment_context,
        }
        
    async def _think_and_tool(self, messages: List[BaseMessage], task_id: str):
        """Execute one ESG analysis step - think and call tools."""
        
        # Get all tools asynchronously and build args_schema dict
        config_tool_names = config.tool_names
        all_tcp_tools = await tcp.list()
        tool_names = list(set(config_tool_names + all_tcp_tools))
        tool_configs = await asyncio.gather(*[tcp.get_info(tool_name) for tool_name in tool_names])
        tcp_args_schema = {
            tool_config.name: tool_config.args_schema for tool_config in tool_configs if tool_config is not None
        }
        agent_args_schema = self.think_output_builder.schemas
        
        logger.info(f"| 📝 ESG Tools: {len(tcp_args_schema)}, Agent Schemas: {len(agent_args_schema)}")
        
        if len(set(tcp_args_schema.keys()) - set(agent_args_schema.keys())) > 0:
            self.think_output_builder.register(tcp_args_schema)
            self.ThinkOutput = self.think_output_builder.build()
        
        done = False
        final_result = None
        
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
                response_format=self.ThinkOutput
            )
            think_output = think_output.extra["parsed_model"]
            
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
            
            logger.info(f"| 💭 ESG Thinking: {thinking[:self.log_max_length]}...")
            logger.info(f"| 🎯 Next ESG Goal: {next_goal}")
            logger.info(f"| 🔧 ESG Tools to execute: {len(tools)}")
            
            # Execute tools sequentially
            tool_results = []
            
            for i, tool in enumerate(tools):
                logger.info(f"| 📝 ESG Tool {i+1}/{len(tools)}: {tool.name}")
                
                tool_name = tool.name
                tool_args = tool.args.model_dump()
                
                logger.info(f"| 📝 Tool Name: {tool_name}, Args: {tool_args}")
                
                input = {
                    "name": tool_name,
                    "input": tool_args
                }
                response = await tcp(**input)
                if isinstance(response, ToolResponse):
                    tool_result = response.message
                    response_extra = response.extra if hasattr(response, 'extra') else None
                else:
                    tool_result = str(response)
                    response_extra = response.get('extra') if isinstance(response, dict) else None
                
                logger.info(f"| ✅ ESG Tool {i+1} completed")
                logger.info(f"| 📄 Results: {str(tool_result)[:self.log_max_length]}...")
                
                # Update tool with result
                tool_dict = tool.model_dump()
                tool_dict["output"] = tool_result
                tool_results.append(tool_dict)
                
                # Update record tool
                tool_extra = {}
                tool_extra.update(tool_dict)
                if response_extra is not None:
                    tool_extra['extra'] = response_extra
                record_tool["tool"].append(tool_extra)
                    
                if tool_name == "done":
                    done = True
                    final_result = tool_result
                    break
            
            event_data = {
                "thinking": thinking,
                "evaluation_previous_goal": evaluation_previous_goal,
                "memory": memory,
                "next_goal": next_goal,
                "tool": tool_results
            }
            
            # Update record tool
            self.record.tool = record_tool
            
            # Get memory system name
            memory_name = self.memory_name
            
            await memory_manager.add_event(
                memory_name=memory_name,
                step_number=self.step_number,
                event_type="tool_step",
                data=event_data,
                agent_name=self.name,
                task_id=task_id
            )
            self.step_number += 1
            
            if done:
                await memory_manager.add_event(
                    memory_name=memory_name,
                    step_number=self.step_number,
                    event_type="task_end",
                    data=dict(result=final_result),
                    agent_name=self.name,
                    task_id=task_id
                )
            
        except Exception as e:
            logger.error(f"| ❌ Error in ESG analysis step: {e}")
        
        return done, final_result
        
    async def __call__(
        self, 
        task: str, 
        files: Optional[List[str]] = None
    ) -> Any:
        """
        Main entry point for ESG Agent.
        
        Args:
            task (str): The ESG analysis task to complete.
            files (Optional[List[str]]): Optional files to attach (e.g., ESG reports).
            
        Returns:
            Any: The final ESG analysis result.
        """
        logger.info(f"| 🌱 Starting ESG Agent: {task}")
        
        if files:
            logger.info(f"| 📂 Attached ESG files: {files}")
            files = await asyncio.gather(*[self._extract_file_content(file) for file in files])
            enhanced_task = await self._generate_enhanced_task(task, files)
        else:
            enhanced_task = task
        
        # Get memory system name
        memory_name = self.memory_name
        
        # Get memory instance to check for restored session
        logger.info(f"| 🔍 Getting ESG memory instance: {memory_name}")
        memory_instance = await memory_manager.get(memory_name)
        logger.info(f"| ✅ Got ESG memory instance: {memory_name}")
        restored_session_id = memory_instance.current_session_id
        logger.info(f"| 🔍 Restored session ID: {restored_session_id}")
        
        if restored_session_id:
            # Restore from checkpoint
            logger.info(f"| 🔄 Restoring ESG session from checkpoint: {restored_session_id}")
            session_id = restored_session_id
            session_info_obj = await memory_manager.get_session_info(memory_name, session_id=session_id)
            description = session_info_obj.description if session_info_obj else None
        else:
            # Start new session
            logger.info(f"| 🆕 Starting new ESG session...")
            session_info = await self._generate_session_info(enhanced_task)
            session_id = session_info.session_id
            description = session_info.description
            logger.info(f"| 📝 ESG Session ID: {session_id}, Description: {description}")
            await memory_manager.start_session(
                memory_name=memory_name,
                session_id=session_id,
                agent_name=self.name,
                description=description
            )
            logger.info(f"| ✅ ESG Session started successfully")
        
        # Add task start event
        task_id = "esg_task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        await memory_manager.add_event(
            memory_name=memory_name,
            step_number=self.step_number,
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
            logger.info(f"| 🔄 ESG Analysis Step {step_number}/{self.max_steps}")
            
            # Execute one step
            done, final_result = await self._think_and_tool(messages, task_id)
            self.step_number += 1
            
            # Update tracer and save to json
            await self.tracer.add_record(
                observation=self.record.observation, 
                tool=self.record.tool,
                session_id=session_id,
                task_id=task_id
            )
            await self.tracer.save_to_json(self.tracer_save_path)
            
            # Memory is automatically saved in add_event()
            messages = await self._get_messages(enhanced_task)
            
            if done:
                break
        
        # Handle max steps reached
        if step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max ESG analysis steps ({self.max_steps}), stopping...")
            final_result = "Reached maximum number of ESG analysis steps"
            
        # Reformulate final result
        reformulator_tool = await tcp.get("reformulator")
        if reformulator_tool:
            data = [
               message.text for message in messages
            ]
            try:
                response = await reformulator_tool(task=task, data=data)
                if response.success:
                    final_result = response.message
                else:
                    logger.error(f"| ❌ Failed to reformulate final result: {response.message}")
                    final_result = "Failed to reformulate final result"
            except Exception as e:
                logger.error(f"| ❌ Error reformulating final result: {e}")
                # Keep the original final_result if reformulation fails
        else:
            logger.warning("| ⚠️ Reformulator tool not found, skipping reformulation")
        
        # Add task end event
        await memory_manager.add_event(
            memory_name=memory_name,
            step_number=self.step_number,
            event_type="task_end",
            data=dict(result=final_result),
            agent_name=self.name,
            task_id=task_id
        )
        
        # End session
        await memory_manager.end_session(memory_name=memory_name, session_id=session_id)
        
        # Save tracer to json
        await self.tracer.save_to_json(self.tracer_save_path)
        
        logger.info(f"| ✅ ESG Agent completed after {step_number}/{self.max_steps} steps")
        
        return final_result

