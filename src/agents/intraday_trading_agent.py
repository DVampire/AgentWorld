"""Intraday trading agent - three-agent architecture in one file.

This file contains three agents:
1. IntradayDayAnalysisAgent - Deep daily trend analysis
2. IntradayMinuteTradingAgent - Fast minute-level trading decisions  
3. IntradayTradingAgent - Orchestrator that coordinates the above two agents
"""

from typing import List, Optional, Type, Dict, Any, Tuple, Union
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

from src.agents.protocol.agent import BaseAgent
from src.logger import logger
from src.utils import dedent
from src.agents.protocol import acp
from src.tools.protocol import tcp
from src.environments.protocol import ecp
from src.infrastructures.memory import SessionInfo, EventType
from src.tools.protocol.types import ToolResponse

class DayAnalysisOutput(BaseModel):
    """Output schema for day analysis."""
    trend_type: str = Field(description="Intraday trend pattern: Uptrend/Downtrend/Up then down/Down then up/Sideways/Volatile")
    confidence: str = Field(description="Confidence level: high/medium/low")
    reasoning: str = Field(description="5-8 sentences explaining the forecast and why this pattern is expected")

class IntradayDayAnalysisAgentInputArgs(BaseModel):
    """Input args for day analysis agent."""
    data: str = Field(description="The data for the day analysis agent.")


class IntradayDayAnalysisAgent(BaseAgent):
    """Intraday day analysis agent - performs deep daily trend analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="intraday_day_analysis", description="The name of the day analysis agent.")
    type: str = Field(default="Agent", description="The type of the day analysis agent.")
    description: str = Field(default="An agent that performs deep daily trend analysis for intraday trading.", description="The description of the day analysis agent.")
    args_schema: Type[IntradayDayAnalysisAgentInputArgs] = Field(default=IntradayDayAnalysisAgentInputArgs, description="The args schema.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata.")
    
    def __init__(
        self,
        workdir: str,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Set default prompt name
        if not prompt_name:
            prompt_name = "intraday_day_analysis"
        
        super().__init__(
            workdir=workdir,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_config=memory_config,
            **kwargs
        )
        
        # Bind model with structured output
        self.structured_model = self.model.with_structured_output(
            DayAnalysisOutput,
            method="function_calling",
            include_raw=False
        )
        
    async def start(self, 
                    task: str, 
                    session_id: str, 
                    task_id: str, 
                    description: str):
        """Start the day analysis agent."""
        # Start session
        await self.memory_manager.start_session(session_id, description)
        await self.memory_manager.add_event(step_number=self.step_number, 
            event_type="task_start", 
            data=dict(task=task),
            agent_name=self.name,
            task_id=task_id
        )
    
    async def end(self, session_id: str):
        """End the day analysis agent."""
        await self.memory_manager.end_session(session_id=session_id)
        
    async def _get_agent_history(self) -> Dict[str, Any]:
        """Get the agent history."""
        state = await self.memory_manager.get_state(n=self.review_steps)
        
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
                agent_history += f"Trend Type: {event.data['trend_type']}\n"
                agent_history += f"Confidence: {event.data['confidence']}\n"
                agent_history += f"Reasoning: {event.data['reasoning']}\n"
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
        
    
    async def _get_environment_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get the environment state."""
        environment_state = data['news_string']
        return {
            "environment_state": environment_state,
        }
    
    async def _get_messages(self, data: Dict[str, Any]) -> List[BaseMessage]:
        """Get messages for the day analysis agent."""
        system_input_variables = {}
        system_message = self.prompt_manager.get_system_message(system_input_variables)
        
        agent_input_variables = {}
        agent_history = await self._get_agent_history()
        environment_state = await self._get_environment_state(data)
        agent_input_variables.update(agent_history)
        agent_input_variables.update(environment_state)
        agent_message = self.prompt_manager.get_agent_message(agent_input_variables)
        
        messages = [
            system_message,
            agent_message,
        ]
        
        return messages
        
    async def ainvoke(self, data: Dict[str, Any], task_id: str):
        """Invoke the day analysis agent."""
        
        messages = await self._get_messages(data)
        
        try:
            response = await self.structured_model.ainvoke(messages)
            
            trend_type = response.trend_type
            confidence = response.confidence
            reasoning = response.reasoning
            
            logger.info(f"| 📝 Trend Type: {trend_type}")
            logger.info(f"| 📝 Confidence: {confidence}")
            logger.info(f"| 📝 Reasoning: {reasoning}")
            
        except Exception as e:
            logger.error(f"| 🚨 Error: {e}")
        
        event_data = {
            "trend_type": trend_type,
            "confidence": confidence,
            "reasoning": reasoning,
        }
        
        await self.memory_manager.add_event(
            step_number=self.step_number,
            event_type="action_step",
            data=event_data,
            agent_name=self.name,
            task_id=task_id
        )
        
        return response


# ============================================
# Agent 2: Minute Trading Agent
# ============================================

class MinuteTradingOutputBuilder:
    def __init__(self):
        self.schemas: Dict[str, type[BaseModel]] = {}

    def register(self, schema: Dict[str, type[BaseModel]]):
        """Register new args schema"""
        self.schemas.update(schema)
        return self  # Support chaining

    def build(self):
        """Generate Action and MinuteTradingOutput models"""

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

        # -------- Dynamically generate MinuteTradingOutput --------
        class MinuteTradingOutput(BaseModel):
            analysis: str = Field(description="2-3 sentences: current price action vs daily trend")
            position_check: str = Field(description="cash/long - duration if holding")
            decision: str = Field(description="BUY/SELL/HOLD")
            reasoning: str = Field(description="Why this decision? Align with daily forecast")
            action: Action = Field(
                description='{"name": "action_name", "args": {...}}'
            )

            def __str__(self):
                return (
                    f"Analysis: {self.analysis}\n"
                    f"Position Check: {self.position_check}\n"
                    f"Decision: {self.decision}\n"
                    f"Reasoning: {self.reasoning}\n"
                    f"Action: {self.action.model_dump()}\n"
                )
            
            def __repr__(self):
                return self.__str__()

        return MinuteTradingOutput


class IntradayMinuteTradingAgentInputArgs(BaseModel):
    """Input args for minute trading agent."""
    data: str = Field(description="The data for the minute trading agent.")


@acp.agent()
class IntradayMinuteTradingAgent(BaseAgent):
    """Intraday minute trading agent - fast execution based on day analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="intraday_minute_trading", description="The name of the minute trading agent.")
    type: str = Field(default="Agent", description="The type of the minute trading agent.")
    description: str = Field(default="An agent that makes fast trading decisions based on daily forecast.", description="The description.")
    args_schema: Type[IntradayMinuteTradingAgentInputArgs] = Field(default=IntradayMinuteTradingAgentInputArgs, description="The args schema.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata.")
    
    def __init__(
        self,
        workdir: str,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Set default prompt name
        if not prompt_name:
            prompt_name = "intraday_minute_trading"
        
        super().__init__(
            workdir=workdir,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_config=memory_config,
            **kwargs
        )
        
        self.minute_trading_output_builder = MinuteTradingOutputBuilder()
        self.minute_trading_output_builder.register(tcp.args_schemas())
        self.MinuteTradingOutput = self.minute_trading_output_builder.build()
        
        # Bind model with structured output
        self.structured_model = self.model.with_structured_output(
            self.MinuteTradingOutput,
            method="function_calling",
            include_raw=False
        )
    
    async def start(self, 
                    task: str, 
                    session_id: str, 
                    task_id: str, 
                    description: str):
        """Start the minute trading agent."""
        # Start session
        await self.memory_manager.start_session(session_id, description)
        await self.memory_manager.add_event(step_number=self.step_number, 
                                      event_type="task_start", 
                                      data=dict(task=task),
                                      agent_name=self.name,
                                      task_id=task_id
                                      )
    
    async def end(self, session_id: str):
        """End the minute trading agent."""
        await self.memory_manager.end_session(session_id=session_id)
    
    async def _get_agent_history(self) -> Dict[str, Any]:
        """Get the agent history."""
        state = await self.memory_manager.get_state(n=self.review_steps)
        
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
                agent_history += f"Analysis: {event.data['analysis']}\n"
                agent_history += f"Position Check: {event.data['position_check']}\n"
                agent_history += f"Decision: {event.data['decision']}\n"
                agent_history += f"Reasoning: {event.data['reasoning']}\n"
                agent_history += f"Action: {event.data['action']}\n"
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
        
    async def _get_agent_state(self) -> Dict[str, Any]:
        """Get the agent state."""
        available_actions_description = [tcp.to_string(tool) for tool in tcp.list()]
        available_actions_description = "\n".join(available_actions_description)
        
        return {
            "available_actions": available_actions_description,
        }
    
    async def _get_environment_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get the environment state."""
        
        environment_state = ""
        info_string = data['info_string']
        environment_state += f"{info_string}\n"
        review_actions_string = data['review_actions_string']
        environment_state += f"{review_actions_string}\n"
        review_trends_string = data['review_trends_string']
        environment_state += f"{review_trends_string}\n"
        
        return {
            "environment_state": environment_state,
        }
    
    async def _get_messages(self, data: Dict[str, Any], daily_trend_forecast: str) -> List[BaseMessage]:
        """Get messages for the minute trading agent."""
        system_input_variables = {}
        system_message = self.prompt_manager.get_system_message(system_input_variables)
        
        agent_input_variables = {}
        agent_history = await self._get_agent_history()
        environment_state = await self._get_environment_state(data)
        agent_input_variables.update(agent_history)
        agent_input_variables.update(environment_state)
        agent_input_variables.update(dict(daily_trend_forecast=daily_trend_forecast))
        agent_message = self.prompt_manager.get_agent_message(agent_input_variables)
        
        messages = [
            system_message,
            agent_message,
        ]
        
        return messages
    
    async def ainvoke(self, data: Dict[str, Any], task_id: str, daily_trend_forecast: str):
        """Invoke the minute trading agent."""
        messages = await self._get_messages(data, daily_trend_forecast)
        
        done = False
        final_result = None
        action_result = None
        
        try:
            minute_trading_output = await self.structured_model.ainvoke(messages)
            
            analysis = minute_trading_output.analysis
            position_check = minute_trading_output.position_check
            decision = minute_trading_output.decision
            reasoning = minute_trading_output.reasoning
            action = minute_trading_output.action
            
            logger.info(f"| 📝 Analysis: {analysis}")
            logger.info(f"| 📝 Position Check: {position_check}")
            logger.info(f"| 📝 Decision: {decision}")
            logger.info(f"| 📝 Reasoning: {reasoning}")
            logger.info(f"| 📝 Action: {action}")
            
            # Execute action
            tool_name = action.name
            tool_args = action.args.model_dump()
            tool_result = await tcp.ainvoke(tool_name, input=tool_args)
            if isinstance(tool_result, ToolResponse):
                tool_result = tool_result.content
            else:
                tool_result = str(tool_result)
            
            logger.info(f"| ✅ Action {tool_name} completed successfully")
            logger.info(f"| 📄 Results: {tool_result}")
            
            # Update action with result
            action_dict = action.model_dump()
            action_dict["output"] = tool_result
            action_result = action_dict
            
            # Check if trading environment is done
            if tool_name == "step" and "Environment status: done" in str(tool_result):
                done = True
                final_result = tool_result
        
        except Exception as e:
            logger.error(f"| 🚨 Error: {e}")
            done = True
            final_result = str(e)
        
        event_data = {
            "analysis": analysis,
            "position_check": position_check,
            "decision": decision,
            "reasoning": reasoning,
            "action": action_result
        }
        
        await self.memory_manager.add_event(
            step_number=self.step_number,
            event_type="action_step",
            data=event_data,
            agent_name=self.name,
            task_id=task_id
        )
        
        return done, final_result


# ============================================
# Agent 3: Main Intraday Trading Agent (Orchestrator)
# ============================================

class IntradayTradingAgentInputArgs(BaseModel):
    """Input args for intraday trading agent."""
    task: str = Field(description="The trading task to complete.")


@acp.agent()
class IntradayTradingAgent(BaseAgent):
    """Intraday trading agent - coordinates day analysis and minute trading agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="intraday_trading", description="The name of the intraday trading agent.")
    type: str = Field(default="Agent", description="The type of the intraday trading agent.")
    description: str = Field(default="An intraday trading agent with two-stage decision making.", description="The description.")
    args_schema: Type[IntradayTradingAgentInputArgs] = Field(default=IntradayTradingAgentInputArgs, description="The args schema.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata.")
    
    def __init__(
        self,
        workdir: str,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        max_steps: int = -1,  # -1 means unlimited
        review_steps: int = 5,
        log_max_length: int = 1000,
        **kwargs
    ):
        super().__init__(
            workdir=workdir,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_config=memory_config,
            max_steps=max_steps,
            review_steps=review_steps,
            log_max_length=log_max_length,
            **kwargs
        )
        
        # Initialize sub-agents
        self.day_analysis_agent = IntradayDayAnalysisAgent(
            workdir=workdir,
            model_name=model_name,
            prompt_name="intraday_day_analysis",
            memory_config=memory_config
        )   
        
        self.minute_trading_agent = IntradayMinuteTradingAgent(
            workdir=workdir,
            model_name=model_name,
            prompt_name="intraday_minute_trading",
            memory_config=memory_config
        )
        
        self.daily_trend_forecast = "No news available. Stay neutral."
        
        logger.info("| 🤖 Intraday Trading Agent initialized with two-stage system")
    
    async def _generate_session_info(self, task: str) -> SessionInfo:
        """Use the llm to generate a session id."""
        structured_llm = self.model.with_structured_output(
            SessionInfo,
            method="function_calling",
            include_raw=False
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful assistant that generates a session info for agent {self.name}."),
            ("user", 
             dedent(f"""
                    <intro>
                    1. The session ID should be a unique identifier for the session that concisely describes the task in snake_case.
                    2. The session description should provide a concise description of the task.
                    </intro>
                    <task>
                    {task}
                    </task>"""
                )
             )
        ])

        chain = prompt | structured_llm
        result: SessionInfo = chain.invoke({"task": task})
        
        timestamp = datetime.now().isoformat()
        
        session_id = f"{self.name}_{timestamp}"
        description = result.description
        
        return SessionInfo(session_id=session_id, description=description)
    
    
    async def _think_and_action(self, data: Dict[str, Any], task_id: str) -> Tuple[bool, Any]:
        """Think and action."""
        
        has_news = data['has_news']
        
        if has_news:
            # Get daily trend forecast
            response = await self.day_analysis_agent.ainvoke(data, task_id)
            daily_trend_forecast = dedent(f"""
                Trend Type: {response.trend_type}
                Confidence: {response.confidence}
                Reasoning: {response.reasoning}
            """)
            logger.info(f"| 📝 Daily Trend Forecast: {daily_trend_forecast}")
            
            self.daily_trend_forecast = daily_trend_forecast
            
        # Get minute trading decision
        done, final_result = await self.minute_trading_agent.ainvoke(data, task_id, self.daily_trend_forecast)
        
        return done, final_result
        
    
    async def ainvoke(
        self, 
        task: str, 
        files: Optional[List[str]] = None,
    ):
        """Run the intraday trading agent with loop."""
        logger.info(f"| 🚀 Starting IntradayTradingAgent: {task}")
        
        session_info = await self._generate_session_info(task)
        session_id = session_info.session_id
        description = session_info.description
        
        # Add task start event
        task_id = "task_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Start
        await self.day_analysis_agent.start(task, session_id, task_id, description)
        await self.minute_trading_agent.start(task, session_id, task_id, description)
        
        # Main loop
        step_number = 0
        done = False
        final_result = None
        
        while not done and (self.max_steps == -1 or step_number < self.max_steps):
            step_number += 1
            logger.info(f"| 🔄 Step {step_number}")
            
            state = await ecp.get_state("intraday_trading")
            data = state["extra"]
            
            # Execute one step
            done, final_result = await self._think_and_action(data, task_id)
            step_number += 1
            
            if done:
                break
        
        # Handle max steps reached
        if self.max_steps > 0 and step_number >= self.max_steps:
            logger.warning(f"| 🛑 Reached max steps ({self.max_steps}), stopping...")
            final_result = "Reached maximum number of steps"
        
        # End session
        await self.day_analysis_agent.end(session_id)
        await self.minute_trading_agent.end(session_id)
        
        logger.info(f"| ✅ Agent completed after {step_number} steps")
        
        return final_result
