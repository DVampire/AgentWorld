"""Online trading agent implementation for multi-stock trading operations."""

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
from src.agent.types import Agent, InputArgs, AgentResponse, AgentExtra, ThinkOutput
from src.tool.types import ToolResponse
from src.memory import memory_manager, EventType
from src.tracer import Tracer, Record
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT

@AGENT.register_module(force=True)
class OnlineTradingAgent(Agent):
    """Online trading agent implementation for multi-stock trading operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="online_trading", description="The name of the online trading agent.")
    description: str = Field(default="A online trading agent that can trade online.", description="The description of the online trading agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the online trading agent.")
    require_grad: bool = Field(default=False, description="Whether the agent requires gradients")
    
    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_modules: Optional[Dict[str, Any]] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        max_tools: int = 10,
        max_steps: int = 20,
        review_steps: int = 5,
        log_max_length: int = 1000,
        require_grad: bool = False,
        **kwargs
    ):
        # Set default prompt name for tool calling
        if not prompt_name:
            prompt_name = "online_trading"
        
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name,
            prompt_name=prompt_name,
            prompt_modules=prompt_modules,
            memory_config=memory_config,
            max_tools=max_tools,
            max_steps=max_steps,
            review_steps=review_steps,
            log_max_length=log_max_length,
            require_grad=require_grad,
            **kwargs)
        
        self.tracer_save_path = os.path.join(self.workdir, "tracer.json")
        
        self.tracer = Tracer()
        self.record = Record()
        
        if os.path.exists(self.tracer_save_path):
            self.tracer.load_from_json(self.tracer_save_path)
            # Get the last record from current session if any exist
            last_record = self.tracer.get_last_record()
            if last_record:
                self.record = last_record
        
        
    async def _get_agent_context(self, task: str) -> Dict[str, Any]:
        """Get the agent context."""
        
        task = f"<task>{task}</task>"
        
        step_info_description = f'Step {self.step_number + 1} of {self.max_steps} max possible steps\n'
        time_str = datetime.now().isoformat()
        step_info_description += f'Current date and time: {time_str}'
        step_info = dedent(f"""
            <step_info>
            {step_info_description}
            </step_info>
        """)
        
        state = await memory_manager.get_state(memory_name=self.memory_name, n=self.review_steps)
        
        events = state["events"]
        summaries = state["summaries"]
        insights = state["insights"]
        
        agent_history = "<agent_history>"
        for event in events:
            agent_history += f"<step_{event.step_number}>\n"
            if event.event_type == EventType.TASK_START:
                agent_history += f"Task Start: {event.data['task']}\n"
            elif event.event_type == EventType.TASK_END:
                agent_history += f"Task End: {event.data['result']}\n"
            elif event.event_type == EventType.TOOL_STEP:
                agent_history += f"Thinking: {event.data['thinking']}\n"
                agent_history += f"Memory: {event.data['memory']}\n"
                agent_history += f"Tool: {event.data['tool']}\n"
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
        
    async def _get_environment_context(self) -> Dict[str, Any]:
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
        
        self.record.observation = record_observation
        
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
        
    async def _get_messages(self, task: str) -> List[BaseMessage]:
        
        system_modules = self.prompt_modules.copy()
        # Infer prompt name from agent's prompt_name
        if self.prompt_name:
            system_prompt_name = f"{self.prompt_name}_system_prompt"
            agent_message_prompt_name = f"{self.prompt_name}_agent_message_prompt"
        else:
            system_prompt_name = "online_trading_system_prompt"
            agent_message_prompt_name = "online_trading_agent_message_prompt"
        
        system_message = await prompt_manager.get_system_message(
            prompt_name=system_prompt_name,
            modules=system_modules, 
            reload=False
        )
        
        agent_message_modules = self.prompt_modules.copy()
        agent_message_modules.update(await self._get_agent_context(task))
        agent_message_modules.update(await self._get_environment_context())
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
        
    async def _think_and_action(self, messages: List[BaseMessage], task_id: str) -> Dict[str, Any]:
        """Think and action for one step."""
        
        done = False
        final_result = None
        final_reasoning = None
        
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
            
            logger.info(f"| 💭 Thinking: {thinking[:self.log_max_length]}...")
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
                    "input": tool_args
                }
                tool_response = await tcp(**input)
                tool_result = tool_response.message
                tool_extra = tool_response.extra if hasattr(tool_response, 'extra') else None
                
                logger.info(f"| ✅ Tool {i+1} completed successfully")
                logger.info(f"| 📄 Results: {str(tool_result)[:self.log_max_length]}...")
                
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
                    final_result = tool_result
                    final_reasoning = tool_extra.data.get('reasoning', None) if tool_extra and tool_extra.data else None
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
            
            await memory_manager.add_event(
                memory_name=self.memory_name,
                step_number=self.step_number,
                event_type=EventType.TOOL_STEP,
                data=event_data,
                agent_name=self.name,
                task_id=task_id
            )
            self.step_number += 1
            
            if done:
                await memory_manager.add_event(
                    memory_name=self.memory_name,
                    step_number=self.step_number,
                    event_type=EventType.TASK_END,
                    data=dict(result=final_result),
                    agent_name=self.name,
                    task_id=task_id
                )
            
        except Exception as e:
            logger.error(f"| Error in thinking and action step: {e}")
        
        result = {
            "done": done,
            "final_result": final_result,
            "final_reasoning": final_reasoning
        }
        return result
        
    async def __call__(self, 
                  task: str, 
                  files: Optional[List[str]] = None
                  ) -> AgentResponse:
        """
        Main entry point for online trading agent through acp.
        
        Args:
            task (str): The task to complete.
            files (Optional[List[str]]): The files to attach to the task.
            
        Returns:
            AgentResponse: The response of the agent.
        """
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
        await memory_manager.start_session(
            memory_name=self.memory_name,
            session_id=session_id,
            agent_name=self.name,
            description=description
        )
        
        # Add task start event
        task_id = "task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        await memory_manager.add_event(
            memory_name=self.memory_name,
            step_number=self.step_number, 
            event_type=EventType.TASK_START, 
            data=dict(task=enhanced_task),
            agent_name=self.name,
            task_id=task_id
        )
        
        # Initialize messages
        messages = await self._get_messages(enhanced_task)
        
        # Main loop
        step_number = 0
        response = None
        
        while step_number < self.max_steps:
            step_number += 1
            logger.info(f"| 🔄 Step {step_number}/{self.max_steps}")
            
            # Execute one step
            response = await self._think_and_action(messages, task_id)
            self.step_number += 1
            
            # Update tracer and save to json
            self.tracer.add_record(observation=self.record.observation, 
                                   tool=self.record.tool,
                                   session_id=session_id,
                                   task_id=task_id)
            self.tracer.save_to_json(self.tracer_save_path)
            
            # Memory is automatically saved in add_event()
            
            messages = await self._get_messages(enhanced_task)
            
            if response.done:
                break
        
        # Handle max steps reached
        if step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping...")
            response = {
                "done": False,
                "final_result": "Reached maximum number of steps",
                "final_reasoning": "Reached the maximum number of steps."
            }
        
        # Add task end event
        await memory_manager.add_event(
            memory_name=self.memory_name,
            step_number=self.step_number,
            event_type=EventType.TASK_END,
            data=response,
            agent_name=self.name,
            task_id=task_id
        )
        
        # End session (automatically saves memory to JSON)
        await memory_manager.end_session(memory_name=self.memory_name, session_id=session_id)
        
        # Save tracer to json
        self.tracer.save_to_json(self.tracer_save_path)
        
        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")
        
        return AgentResponse(
            success=response["done"],
            message=response["final_result"] if response["final_result"] else "",
            extra=AgentExtra(data=response)
        )