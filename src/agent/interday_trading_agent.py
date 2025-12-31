"""Interday trading agent implementation for single stock trading tasks."""

from typing import List, Optional, Type, Dict, Any, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from src.agent.types import Agent, ThinkOutputBuilder, InputArgs, AgentResponse, AgentExtra
from src.logger import logger
from src.utils import dedent
from src.agent.server import acp
from src.tool.server import tcp
from src.environment.server import ecp
from src.memory import memory_manager, SessionInfo, EventType
from src.tool.types import ToolResponse
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT

@AGENT.register_module(force=True)
class InterdayTradingAgent(Agent):
    """Interday trading agent implementation for single stock trading tasks."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="interday_trading", description="The name of the interday trading agent.")
    description: str = Field(default="A interday trading agent that can perform single stock trading tasks.", description="The description of the interday trading agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the interday trading agent.")
    require_grad: bool = Field(default=False, description="Whether the agent requires gradients")
    
    def __init__(
        self,
        workdir: str,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        max_steps: int = -1,  # -1 means unlimited steps for trading
        review_steps: int = 5,
        log_max_length: int = 1000,
        require_grad: bool = False,
        **kwargs
    ):
        # Set default prompt name for interday trading
        if not prompt_name:
            prompt_name = "interday_trading"
        
        super().__init__(
            workdir=workdir,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_config=memory_config,
            max_steps=max_steps,
            review_steps=review_steps,
            log_max_length=log_max_length,
            require_grad=require_grad,
            **kwargs)
        
        self.think_output_builder = ThinkOutputBuilder()
        self.think_output_builder.register(tcp.args_schemas())
        self.ThinkOutput = self.think_output_builder.build()
        
    async def _think_and_action(self, messages: List[BaseMessage], task_id: str) -> Dict[str, Any]:
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
        final_reasoning = None
        
        try:
            model_response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=self.ThinkOutput
            )
            think_output = model_response.extra.parsed_model
            
            thinking = think_output.thinking
            evaluation_previous_goal = think_output.evaluation_previous_goal
            memory = think_output.memory
            next_goal = think_output.next_goal
            actions = think_output.action
            
            logger.info(f"| 💭 Thinking: {thinking}...")
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
                
                input = {
                    "name": tool_name,
                    "input": tool_args
                }
                tool_response = await tcp(**input)
                tool_result = tool_response.message
                tool_extra = tool_response.extra if hasattr(tool_response, 'extra') else None
                
                logger.info(f"| ✅ Action {i+1} completed successfully")
                logger.info(f"| 📄 Results: {str(tool_result)}...")
                
                # Update action with result
                action_dict = action.model_dump()
                action_dict["output"] = tool_result
                action_results.append(action_dict)
                    
                # Check if trading environment is done
                if tool_name == "step" and "Environment status: done" in str(tool_result):
                    done = True
                    final_result = tool_result
                    final_reasoning = tool_extra.data.get('reasoning', None) if tool_extra and tool_extra.data else None
                    break
            
            event_data = {
                "thinking": thinking,
                "evaluation_previous_goal": evaluation_previous_goal,
                "memory": memory,
                "next_goal": next_goal,
                "action": action_results
            }
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
    
    async def _generate_session_info(self, task: str) -> SessionInfo:
        """Use the llm to generate a session id."""
        from langchain_core.messages import SystemMessage, HumanMessage
        
        system_prompt = f"You are a helpful assistant that generates a session info for agent {self.name}."
        user_prompt = dedent(f"""
            <intro>
            1. The session ID should be a unique identifier for the session that concisely describes the task in snake_case.
            2. The session description should provide a concise description of the task.
            </intro>
            <task>
            {task}
            </task>"""
        )
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        model_response = await model_manager(
            model=self.model_name,
            messages=messages,
            response_format=SessionInfo
        )
        result = model_response.extra.parsed_model
        
        timestamp = datetime.now().isoformat()
        
        session_id = f"{self.name}_{timestamp}"
        description = result.description
        
        return SessionInfo(session_id=session_id, description=description)
    
    async def _get_agent_history(self) -> Dict[str, Any]:
        """Get the agent history."""
        state = await memory_manager.get_state(memory_name=self.memory_name, n=self.review_steps)
        
        events = state["events"]
        summaries = state["summaries"]
        insights = state["insights"]
        
        agent_history = ""
        for event in events:
            agent_history += f"<step_{event.step_number}>\n"
            if event.event_type == EventType.TASK_START:
                agent_history += f"Task Start: {event.data['task']}\n"
            elif event.event_type == EventType.TASK_END:
                agent_history += f"Task End: {event.data['result']}\n"
            elif event.event_type == EventType.ACTION_STEP:
                agent_history += f"Evaluation of Previous Step: {event.data['evaluation_previous_goal']}\n"
                agent_history += f"Memory: {event.data['memory']}\n"
                agent_history += f"Next Goal: {event.data['next_goal']}\n"
                agent_history += f"Action Results: {event.data['action']}\n"
            agent_history += "\n"
            agent_history += f"</step_{event.step_number}>\n"
        
        agent_history += dedent(f"""
            <summaries>
            {chr(10).join([str(summary) for summary in summaries])}
            </summaries>
            <insights>
            {chr(10).join([str(insight) for insight in insights])}
            </insights>
        """)
        
        return {
            "agent_history": agent_history,
        }
    
    async def _get_agent_state(self, task: str) -> Dict[str, Any]:
        """Get the agent state."""
        step_info_description = f'Step {self.step_number + 1}'
        if self.max_steps > 0:
            step_info_description += f' of {self.max_steps} max possible steps'
        step_info_description += '\n'
        time_str = datetime.now().isoformat()
        step_info_description += f'Current date and time: {time_str}'
        
        available_actions_description = [tcp.to_string(tool) for tool in tcp.list()]
        available_actions_description = "\n".join(available_actions_description)
        
        return {
            "task": task,
            "step_info": step_info_description,
            "available_actions": available_actions_description,
        }
        
    async def _get_environment_state(self) -> Dict[str, Any]:
        """Get the environment state."""
        environment_state = ""
        for env_name in ecp.list():
            env_state = await ecp.get_state(env_name)
            state_string = env_state["state"]
            extra = env_state["extra"]
            
            if "screenshots" in extra:
                for screenshot in extra["screenshots"]:
                    state_string += f"\n<img src={screenshot.screenshot_path} alt={screenshot.screenshot_description}/>"
                    
            environment_state += dedent(f"""
                <{env_name}_state>
                {state_string}
                </{env_name}_state>
            """)
        
        return {
            "environment_state": environment_state,
        }
        
    async def _get_messages(self, task: str) -> List[BaseMessage]:
        
        system_modules = self.prompt_modules.copy()
        # Infer prompt name from agent's prompt_name
        if self.prompt_name:
            system_prompt_name = f"{self.prompt_name}_system_prompt"
            agent_message_prompt_name = f"{self.prompt_name}_agent_message_prompt"
        else:
            system_prompt_name = "interday_trading_system_prompt"
            agent_message_prompt_name = "interday_trading_agent_message_prompt"
        
        # Add environment rules to system modules
        environment_rules = ""
        for env_name in ecp.list():
            environment_rules += f"{ecp.get_info(env_name).rules}\n"
        system_modules.update(dict(
            environment_rules=environment_rules,
        ))
        
        system_message = await prompt_manager.get_system_message(
            prompt_name=system_prompt_name,
            modules=system_modules, 
            reload=False
        )
        
        agent_message_modules = self.prompt_modules.copy()
        agent_history = await self._get_agent_history()
        agent_state = await self._get_agent_state(task)
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
        
    async def __call__(self, 
                  task: str, 
                  files: Optional[List[str]] = None
                  ) -> AgentResponse:
        """
        Main entry point for interday trading agent through acp.
        
        Args:
            task (str): The task to complete.
            files (Optional[List[str]]): The files to attach to the task.
            
        Returns:
            AgentResponse: The response of the agent.
        """
        logger.info(f"| 🚀 Starting InterdayTradingAgent: {task}")
        
        session_info = await self._generate_session_info(task)
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
            data=dict(task=task),
            agent_name=self.name,
            task_id=task_id
        )
        
        # Initialize messages
        messages = await self._get_messages(task)
        
        # Main loop
        step_number = 0
        response = None
        
        while self.max_steps == -1 or step_number < self.max_steps:
            step_number += 1
            logger.info(f"| 🔄 Step {step_number}")
            
            # Execute one step
            response = await self._think_and_action(messages, task_id)
            self.step_number += 1
            
            messages = await self._get_messages(task)
            
            if response.done:
                break
        
        # Handle max steps reached
        if self.max_steps > 0 and step_number >= self.max_steps:
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
        
        # End session
        await memory_manager.end_session(memory_name=self.memory_name, session_id=session_id)
        
        logger.info(f"| ✅ Agent completed after {step_number} steps")
        
        return AgentResponse(
            success=response["done"],
            message=response["final_result"] if response["final_result"] else "",
            extra=AgentExtra(data=response)
        )
