"""Interday trading agent implementation for single stock trading tasks."""

from typing import List, Optional, Type, Dict, Any, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from src.agent.types import Agent, InputArgs, AgentResponse, AgentExtra, ThinkOutput
from src.logger import logger
from src.utils import dedent
from src.agent.server import acp
from src.tool.server import tcp
from src.environment.server import ecp
from src.memory import memory_manager, EventType
from src.tool.types import ToolResponse
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT
from src.session import SessionContext

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
            require_grad=require_grad,
            **kwargs)
        
        
    async def _think_and_action(self, messages: List[BaseMessage], task_id: str, ctx: SessionContext = None) -> Dict[str, Any]:
        """Think and action for one step."""
        
        step_number = ctx.step_number if ctx else None
        
        done = False
        result = None
        reasoning = None
        
        current_step = step_number if step_number is not None else self.step_number
        
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
            actions = think_output.actions
            
            logger.info(f"| 💭 Thinking: {thinking}...")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 🔧 Actions to execute: {len(actions)}")
            
            action_results = []
            
            for i, action in enumerate(actions):
                action_name = action.name
                action_args = action.args if action.args else {}
                
                logger.info(f"| 📝 Action {i+1}/{len(actions)}: {action_name}")
                logger.info(f"| 📝 Args: {action_args}")
                
                input = {
                    "name": action_name,
                    "input": action_args,
                    "ctx": ctx
                }
                action_response = await tcp(**input)
                action_result = action_response.message
                action_extra = action_response.extra if hasattr(action_response, 'extra') else None
                
                logger.info(f"| ✅ Action {i+1} completed successfully")
                logger.info(f"| 📄 Results: {str(action_result)}...")
                
                action_dict = action.model_dump()
                action_dict["output"] = action_result
                action_results.append(action_dict)
                    
                if action_name == "step" and "Environment status: done" in str(action_result):
                    done = True
                    result = action_result
                    reasoning = action_extra.data.get('reasoning', None) if action_extra and action_extra.data else None
                    break
            
            event_data = {
                "thinking": thinking,
                "evaluation_previous_goal": evaluation_previous_goal,
                "memory": memory,
                "next_goal": next_goal,
                "actions": action_results
            }
            await memory_manager.add_event(
                memory_name=self.memory_name,
                step_number=current_step,
                event_type=EventType.TOOL_STEP,
                data=event_data,
                agent_name=self.name,
                task_id=task_id,
                ctx=ctx
            )
            
        except Exception as e:
            logger.error(f"| Error in thinking and action step: {e}")
        
        response_dict = {
            "done": done,
            "result": result,
            "reasoning": reasoning
        }
        return response_dict
    
    async def _get_agent_history(self, ctx: SessionContext = None) -> Dict[str, Any]:
        """Get the agent history."""
        state = await memory_manager.get_state(memory_name=self.memory_name, n=self.review_steps, ctx=ctx)
        
        events = state["events"]
        summaries = state["summaries"]
        insights = state["insights"]
        
        agent_history = ""
        for event in events:
            agent_history += f"<step_{event.step_number}>\n"
            if event.event_type == EventType.TASK_START:
                agent_history += f"Task Start: {event.data.get('task', event.data.get('message', ''))}\n"
            elif event.event_type == EventType.TASK_END:
                agent_history += f"Task End: {event.data.get('result', '')}\n"
            elif event.event_type == EventType.TOOL_STEP:
                agent_history += f"Evaluation of Previous Step: {event.data.get('evaluation_previous_goal', '')}\n"
                agent_history += f"Memory: {event.data.get('memory', '')}\n"
                agent_history += f"Next Goal: {event.data.get('next_goal', '')}\n"
                agent_history += f"Action Results: {event.data.get('actions', event.data.get('tool', ''))}\n"
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
    
    async def _get_agent_state(self, task: str, step_number: Optional[int] = None) -> Dict[str, Any]:
        """Get the agent state."""
        current_step = step_number if step_number is not None else self.step_number
        step_info_description = f'Step {current_step + 1}'
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
        
    async def _get_messages(self, task: str, ctx: SessionContext = None, **kwargs) -> List[BaseMessage]:
        
        step_number = ctx.step_number if ctx else None
        
        system_modules = {}
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
        
        agent_message_modules = {}
        agent_history = await self._get_agent_history(ctx=ctx)
        agent_state = await self._get_agent_state(task, step_number=step_number)
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
                  files: Optional[List[str]] = None,
                  **kwargs
                  ) -> AgentResponse:
        """
        Main entry point for interday trading agent through acp.
        """
        logger.info(f"| 🚀 Starting InterdayTradingAgent: {task}")
        
        # Get id from ctx
        ctx = kwargs.get("ctx", None)
        if ctx is None:
            ctx = SessionContext()
        task_id = "task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        logger.info(f"| 📝 Context ID: {ctx.id}, Task ID: {task_id}")
        
        # Start session
        await memory_manager.start_session(memory_name=self.memory_name, ctx=ctx)
        
        # Add task start event
        await memory_manager.add_event(
            memory_name=self.memory_name,
            step_number=0, 
            event_type=EventType.TASK_START, 
            data=dict(task=task),
            agent_name=self.name,
            task_id=task_id,
            ctx=ctx
        )
        
        # Initialize messages
        ctx.step_number = 0
        messages = await self._get_messages(task, ctx=ctx)
        
        # Main loop
        step_number = 0
        response = None
        
        while self.max_steps == -1 or step_number < self.max_steps:
            logger.info(f"| 🔄 Step {step_number+1}")
            
            ctx.step_number = step_number
            
            # Execute one step
            response = await self._think_and_action(messages, task_id, ctx=ctx)
            step_number += 1
            
            ctx.step_number = step_number
            messages = await self._get_messages(task, ctx=ctx)
            
            if response["done"]:
                break
        
        # Handle max steps reached
        if self.max_steps > 0 and step_number >= self.max_steps:
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
            ctx=ctx
        )
        
        # End session
        await memory_manager.end_session(memory_name=self.memory_name, ctx=ctx)
        
        logger.info(f"| ✅ Agent completed after {step_number} steps")
        
        return AgentResponse(
            success=response["done"],
            message=response["result"] if response["result"] else "",
            extra=AgentExtra(data=response)
        )
