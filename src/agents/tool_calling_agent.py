"""Tool calling agent implementation with manual agent logic."""

import asyncio
import os
from typing import List, Optional, Type, Dict, Any
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import Field, ConfigDict

from src.agents.protocol.agent import BaseAgent, ThinkOutputBuilder
from src.logger import logger
from src.utils import dedent
from src.agents.protocol import acp
from src.tools.protocol import tcp
from src.environments.protocol import ecp
from src.tools.protocol.types import ToolResponse
from src.agents.protocol.types import InputArgs
from src.tracer import Tracer, Record

@acp.agent()
class ToolCallingAgent(BaseAgent):
    """Tool calling agent implementation with manual agent logic."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="tool_calling", description="The name of the tool calling agent.")
    type: str = Field(default="Agent", description="The type of the tool calling agent.")
    description: str = Field(default="A tool calling agent that can call tools to complete tasks.", description="The description of the tool calling agent.")
    args_schema: Type[InputArgs] = Field(default=InputArgs, description="The args schema of the tool calling agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool calling agent.")
    
    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        type: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[InputArgs]] = None,
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
            prompt_name = "tool_calling"
        
        super().__init__(
            workdir=workdir,
            name=name,
            type=type,
            description=description,
            args_schema=args_schema,
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
        
        # Bind tools to model
        self.tools = [tcp.get(tool) for tool in tcp.list()]
        self.no_fc_model = self.model.bind_tools(tools=self.tools, tool_choice="none")
        self.fc_model = self.model.bind_tools(tools=self.tools, tool_choice="any")
    
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
        
    async def _think_and_action(self, messages: List[BaseMessage], task_id: str):
        """Think and action for one step."""
        
        # If the new tool is added, rebuild the ThinkOutput model
        tcp_args_schema = tcp.args_schemas()
        agent_args_schema = self.think_output_builder.schemas
        
        logger.info(f"| 📝 TCP Args Schema: {len(tcp_args_schema)}, Agent Args Schema: {len(agent_args_schema)}")
        
        if len(set(tcp_args_schema.keys()) - set(agent_args_schema.keys())) > 0:
            self.think_output_builder.register(tcp_args_schema)
            self.ThinkOutput = self.think_output_builder.build()
        
        # Get structured output for thinking
        structured_llm = self.no_fc_model.with_structured_output(
            self.ThinkOutput,
            method="function_calling",
            include_raw=False
        )
        
        done = False
        final_result = None
        
        record_action = {
            "thinking": None,
            "evaluation_previous_goal": None,
            "memory": None,
            "next_goal": None,
            "action": [],
        }
        
        try:
            think_output = await structured_llm.ainvoke(messages)
            
            thinking = think_output.thinking
            evaluation_previous_goal = think_output.evaluation_previous_goal
            memory = think_output.memory
            next_goal = think_output.next_goal
            actions = think_output.action
            
            # Update record action
            record_action["thinking"] = thinking
            record_action["evaluation_previous_goal"] = evaluation_previous_goal
            record_action["memory"] = memory
            record_action["next_goal"] = next_goal
            
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
                
                response = await tcp.ainvoke(tool_name, input=tool_args)
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
                "evaluation_previous_goal": evaluation_previous_goal,
                "memory": memory,
                "next_goal": next_goal,
                "action": action_results
            }
            
            # Update record action
            self.record.action = record_action
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
        
        # Check if we should restore from checkpoint
        restored_session_id = self.memory_manager.get_current_session_id()
        
        if restored_session_id:
            # Restore from checkpoint
            logger.info(f"| 🔄 Restoring session from checkpoint: {restored_session_id}")
            session_id = restored_session_id
            session_info_obj = await self.memory_manager.get_session_info(session_id)
            description = session_info_obj.description if session_info_obj else None
        else:
            # Start new session
            session_info = await self._generate_session_info(enhanced_task)
            session_id = session_info.session_id
            description = session_info.description
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
            
            # Update tracer and save to json
            self.tracer.add_record(observation=self.record.observation, 
                                   action=self.record.action,
                                   session_id=session_id,
                                   task_id=task_id)
            self.tracer.save_to_json(self.tracer_save_path)
            
            # Save memory to json
            await self.memory_manager.save_to_json(self.memory_save_path)
            
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
        
        # Save tracer to json
        self.tracer.save_to_json(self.tracer_save_path)
        
        # Save memory to json
        await self.memory_manager.save_to_json(self.memory_save_path)
        
        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")
        
        return final_result