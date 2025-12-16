"""Online trading agent implementation for multi-stock trading operations."""

import asyncio
import os
import pandas as pd
from datetime import datetime
from typing import List, Optional, Type, Dict, Any, Union
from langchain_core.messages import BaseMessage
from pydantic import Field, ConfigDict, BaseModel

from src.logger import logger
from src.utils import dedent
from src.agent.server import acp
from src.tool.server import tcp
from src.environment.server import ecp
from src.agent.types import Agent, ThinkOutputBuilder, InputArgs
from src.tool.types import ToolResponse
from src.memory import memory_manager, EventType
from src.tracer import Tracer, Record
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT

def format_actions(actions: List[BaseModel]) -> str:
    """Format actions as a Markdown table using pandas."""
    rows = []
    for action in actions:
        if isinstance(action.args, dict):
            args_str = ", ".join(f"{k}={v}" for k, v in action.args.items())
        else:
            args_str = str(action.args)

        rows.append({
            "Action": action.name,
            "Args": args_str,
            "Output": action.output if action.output is not None else None
        })
    
    df = pd.DataFrame(rows)
    
    if df["Output"].isna().all():
        df = df.drop(columns=["Output"])
    else:
        df["Output"] = df["Output"].fillna("None")
    
    return df.to_markdown(index=True)

class ThinkOutputBuilder:
    def __init__(self):
        self.schemas: Dict[str, type[BaseModel]] = {}

    def register(self, schema: Dict[str, type[BaseModel]]):
        """Register new args schema"""
        self.schemas.update(schema)
        return self  # Support chaining

    def build(self):
        """Generate Action and ThinkOutput models"""

        # -------- Dynamically generate Action --------
        schemas = self.schemas
        ActionArgs = Union[tuple(schemas.values())]

        class Action(BaseModel):
            name: str = Field(description="The name of the action.")
            args: ActionArgs = Field(description="The arguments of the action.")
            output: Optional[str] = Field(default=None, description="The output of the action.")
            
            def __str__(self):
                return f"Action: {self.name}\nArgs: {self.args}\nOutput: {self.output}\n"
            
            def __repr__(self):
                return self.__str__()

        # -------- Dynamically generate ThinkOutput --------
        class ThinkOutput(BaseModel):
            thinking: str = Field(description="A structured <think>-style reasoning block.")
            memory: str = Field(description="1-3 sentences of specific memory.")
            action: List[Action] = Field(
                description='[{"name": "action_name", "args": {...}}, ...]'
            )

            def __str__(self):
                return (
                    f"Thinking: {self.thinking}\n"
                    f"Memory: {self.memory}\n"
                    f"Action:\n{format_actions(self.action)}\n"
                )
            
            def __repr__(self):
                return self.__str__()

        return ThinkOutput

@AGENT.register_module(force=True)
class OnlineTradingAgent(Agent):
    """Online trading agent implementation for multi-stock trading operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="online_trading", description="The name of the online trading agent.")
    description: str = Field(default="A online trading agent that can trade online.", description="The description of the online trading agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the online trading agent.")
    
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
        
        self.think_output_builder = ThinkOutputBuilder()
        self.think_output_builder.register(tcp.args_schemas())
        self.ThinkOutput = self.think_output_builder.build()
        
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
            elif event.event_type == EventType.ACTION_STEP:
                agent_history += f"Thinking: {event.data['thinking']}\n"
                agent_history += f"Memory: {event.data['memory']}\n"
                agent_history += f"Action: {event.data['action']}\n"
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
        
        for env_name in ecp.list():
            rule_string = ecp.get_info(env_name).rules
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
        
    async def _think_and_action(self, messages: List[BaseMessage], task_id: str):
        """Think and action for one step."""
        
        # If the new tool is added, rebuild the ThinkOutput model
        tcp_args_schema = tcp.args_schemas()
        agent_args_schema = self.think_output_builder.schemas
        
        logger.info(f"| 📝 TCP Args Schema: {len(tcp_args_schema)}, Agent Args Schema: {len(agent_args_schema)}")
        
        if len(set(tcp_args_schema.keys()) - set(agent_args_schema.keys())) > 0:
            self.think_output_builder.register(tcp_args_schema)
            self.ThinkOutput = self.think_output_builder.build()
        
        done = False
        final_result = None
        
        record_action = {
            "thinking": None,
            "memory": None,
            "action": [],
        }
        
        try:
            model_response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=self.ThinkOutput
            )
            think_output = model_response.extra["parsed_model"]
            
            thinking = think_output.thinking
            memory = think_output.memory
            actions = think_output.action
            
            # Update record action
            record_action["thinking"] = thinking
            record_action["memory"] = memory
            
            logger.info(f"| 💭 Thinking: {thinking[:self.log_max_length]}...")
            logger.info(f"| 🔧 Actions to execute: {len(actions)}")
            
            # Execute actions sequentially
            action_results = []
            
            for i, action in enumerate(actions):
                logger.info(f"| 📝 Action {i+1}/{len(actions)}: {action.name}")
                
                # Execute the tool
                tool_name = action.name
                tool_args = action.args.model_dump()
                
                logger.info(f"| 📝 Action Name: {tool_name}, Args: {tool_args}")
                
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
                
                logger.info(f"| ✅ Action {i+1} completed successfully")
                logger.info(f"| 📄 Results: {str(tool_result)[:self.log_max_length]}...")
                
                # Update action with result
                action_dict = action.model_dump()
                action_dict["output"] = tool_result
                action_results.append(action_dict)
                
                # Update record action
                action_extra = {}
                action_extra.update(action_dict)
                if response_extra is not None:
                    action_extra['extra'] = response_extra
                record_action["action"].append(action_extra)
                    
                if tool_name == "done":
                    done = True
                    final_result = tool_result
                    break
            
            event_data = {
                "thinking": thinking,
                "memory": memory,
                "action": action_results
            }
            
            # Update record action
            self.record.action = record_action
            
            await memory_manager.add_event(
                memory_name=self.memory_name,
                step_number=self.step_number,
                event_type="action_step",
                data=event_data,
                agent_name=self.name,
                task_id=task_id
            )
            self.step_number += 1
            
            if done:
                await memory_manager.add_event(
                    memory_name=self.memory_name,
                    step_number=self.step_number,
                    event_type="task_end",
                    data=dict(result=final_result),
                    agent_name=self.name,
                    task_id=task_id
                )
            
        except Exception as e:
            logger.error(f"| Error in thinking and action step: {e}")
        
        return done, final_result
        
    async def __call__(self, 
                  task: str, 
                  files: Optional[List[str]] = None
                  ) -> Any:
        """
        Main entry point for online trading agent through acp.
        
        Args:
            task (str): The task to complete.
            files (Optional[List[str]]): The files to attach to the task.
            
        Returns:
            Any: The final result of the task.
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
            
            # Update tracer and save to json
            self.tracer.add_record(observation=self.record.observation, 
                                   action=self.record.action,
                                   session_id=session_id,
                                   task_id=task_id)
            self.tracer.save_to_json(self.tracer_save_path)
            
            # Memory is automatically saved in add_event()
            
            messages = await self._get_messages(enhanced_task)
            
            if done:
                break
        
        # Handle max steps reached
        if step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping...")
            final_result = "Reached maximum number of steps"
        
        # Add task end event
        await memory_manager.add_event(
            memory_name=self.memory_name,
            step_number=self.step_number,
            event_type="task_end",
            data=dict(result=final_result),
            agent_name=self.name,
            task_id=task_id
        )
        
        # End session (automatically saves memory to JSON)
        await memory_manager.end_session(memory_name=self.memory_name, session_id=session_id)
        
        # Save tracer to json
        self.tracer.save_to_json(self.tracer_save_path)
        
        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")
        
        return final_result