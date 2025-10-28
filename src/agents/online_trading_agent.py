"""Online trading agent implementation for multi-stock trading operations."""

import asyncio
from typing import List, Optional, Type, Dict, Any
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import Field, ConfigDict

from src.agents.protocol.agent import BaseAgent, ThinkOutputBuilder
from src.logger import logger
from src.agents.protocol import acp
from src.tools.protocol import tcp
from src.tools.protocol.types import ToolResponse
from src.agents.protocol.types import InputArgs

@acp.agent()
class OnlineTradingAgent(BaseAgent):
    """Online trading agent implementation for multi-stock trading operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="online_trading", description="The name of the online trading agent.")
    type: str = Field(default="Agent", description="The type of the online trading agent.")
    description: str = Field(default="A online trading agent that can trade online.", description="The description of the online trading agent.")
    args_schema: Type[InputArgs] = Field(default=InputArgs, description="The args schema of the online trading agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the online trading agent.")
    
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
            prompt_name = "online_trading"
        
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
        
        self.think_output_builder = ThinkOutputBuilder()
        self.think_output_builder.register(tcp.args_schemas())
        self.ThinkOutput = self.think_output_builder.build()
        
        # Bind tools to model
        self.tools = [tcp.get(tool) for tool in tcp.list()]
        self.no_fc_model = self.model.bind_tools(tools=self.tools, tool_choice="none")
        self.fc_model = self.model.bind_tools(tools=self.tools, tool_choice="any")
        
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
        
        try:
            think_output = await structured_llm.ainvoke(messages)
            
            thinking = think_output.thinking
            evaluation_previous_goal = think_output.evaluation_previous_goal
            memory = think_output.memory
            next_goal = think_output.next_goal
            actions = think_output.action
            
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
                
                tool_result = await tcp.ainvoke(tool_name, input=tool_args)
                if isinstance(tool_result, ToolResponse):
                    tool_result = tool_result.content
                else:
                    tool_result = str(tool_result)
                
                logger.info(f"| ✅ Action {i+1} completed successfully")
                logger.info(f"| 📄 Results: {str(tool_result)[:self.log_max_length]}...")
                
                # Update action with result
                action_dict = action.model_dump()
                action_dict["output"] = tool_result
                action_results.append(action_dict)
                    
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
        
        session_info = await self._generate_session_info(enhanced_task)
        session_id = session_info.session_id
        description = session_info.description
        
        # Start session
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
        
        logger.info(f"| ✅ Agent completed after {step_number}/{self.max_steps} steps")
        
        return final_result