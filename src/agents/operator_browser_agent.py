"""Operator Browser Agent implementation for web automation tasks using vision-enabled LLM."""

import asyncio
from typing import List, Optional, Type, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
import base64

from src.agents.protocol.agent import BaseAgent, ThinkOutputBuilder
from src.logger import logger
from src.infrastructures.models import model_manager
from src.utils import get_file_info, dedent
from src.agents.protocol import acp
from src.tools.protocol import tcp
from src.environments.protocol import ecp
from src.infrastructures.memory import SessionInfo, EventType
from src.tools.protocol.types import ToolResponse
from src.agents.prompts.prompt_manager import PromptManager

class OperatorBrowserAgentInputArgs(BaseModel):
    task: str = Field(description="The web browsing task to complete.")


@acp.agent()
class OperatorBrowserAgent(BaseAgent):
    """Operator Browser Agent implementation with visual understanding capabilities."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="operator_browser", description="The name of the operator browser agent.")
    type: str = Field(default="Agent", description="The type of the operator browser agent.")
    description: str = Field(default="A browser operator agent that can see and control web browsers using vision-enabled LLM.", description="The description of the operator browser agent.")
    args_schema: Type[OperatorBrowserAgentInputArgs] = Field(default=OperatorBrowserAgentInputArgs, description="The args schema of the operator browser agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the operator browser agent.")
    
    def __init__(
        self,
        workdir: str,
        model_name: Optional[str] = "gpt-4.1",
        prompt_name: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        max_steps: int = 30,
        review_steps: int = 5,
        log_max_length: int = 500,
        if_correct_action: bool = True,
        **kwargs
    ):
        """Initialize the Operator Browser Agent.
        
        Args:
            workdir: Working directory for logs and screenshots
            model_name: LLM model name (should support vision, default: computer-browser-use)
            prompt_name: Name of the prompt template (default: operator_browser)
            memory_config: Memory configuration
            max_steps: Maximum number of steps
            review_steps: Number of steps to review in history
            log_max_length: Maximum log length
            if_correct_action: If True, correct the action
        """
        # Set default prompt name for operator browser
        if not prompt_name:
            prompt_name = "operator_browser"
        
        super().__init__(
            workdir=workdir,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_config=memory_config,
            max_steps=max_steps,
            review_steps=review_steps,
            log_max_length=log_max_length,
            **kwargs)
        
        self.if_correct_action = if_correct_action
        
        self.prompt_manager = PromptManager(
            prompt_name=prompt_name,
            max_actions_per_step=1, # Max actions per step is 1 for operator browser agent
        )
        
        self.think_output_builder = ThinkOutputBuilder()
        self.think_output_builder.register(tcp.args_schemas())
        self.ThinkOutput = self.think_output_builder.build()
        
        # Bind tools to model
        self.tools = [tcp.get(tool) for tool in tcp.list()]
        self.no_fc_model = self.model.bind_tools(tools=self.tools, tool_choice="none")
        self.fc_model = self.model.bind_tools(tools=self.tools, tool_choice="any")
        
        self.cua_model = model_manager.get(model_name="computer-browser-use")
        self.cua_prompt_manager = PromptManager(
            prompt_name="operator_browser_cua",
        )
    
    async def _extract_file_content(self, file: str) -> str:
        """Extract file information."""
        
        info = get_file_info(file)
        
        # Extract file content
        file_content = await tcp.ainvoke("mdify", input={"file_path": file, "output_format": "markdown"})
        
        # Use LLM to summarize the file content
        system_prompt = "You are a helpful assistant that summarizes file content."
        
        user_prompt = dedent(f"""
            Summarize the following file content as 1-3 sentences:
            {file_content}
        """)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        summary = await self.no_fc_model.ainvoke(messages)
        
        info["content"] = file_content
        info["summary"] = summary.content
        
        return info
    
    async def _generate_enhanced_task(self, task: str, files: List[str]) -> str:
        """Generate enhanced task."""
        
        attach_files_string = "\n".join([f"File: {file['path']}\nSummary: {file['summary']}" for file in files])
        
        enhanced_task = dedent(f"""
        - Task:
        {task}
        - Attach files:
        {attach_files_string}
        """)
        return enhanced_task
    
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
                agent_history += f"Evaluation of Previous Step: {event.data['evaluation_previous_goal']}\n"
                agent_history += f"Memory: {event.data['memory']}\n"
                agent_history += f"Next Goal: {event.data['next_goal']}\n"
                agent_history += f"Action Results: {event.data['action']}\n"
            agent_history += "\n"
            agent_history += f"</step_{event.step_number}>\n"
        
        if len(summaries) > 0:
            agent_history += dedent(f"""
                <summaries>
                {chr(10).join([str(summary) for summary in summaries])}
                </summaries>
            """)
        if len(insights) > 0:
            agent_history += dedent(f"""
                <insights>
                {chr(10).join([str(insight) for insight in insights])}
                </insights>
            """)
        
        return {
            "agent_history": agent_history,
        }
    
    async def _get_todo_contents(self) -> str:
        """Get the todo contents."""
        todo_tool = tcp.get("todo")
        todo_contents = todo_tool.get_todo_content()
        return todo_contents   
    
    async def _get_agent_state(self, task: str) -> Dict[str, Any]:
        """Get the agent state."""
        step_info_description = f'Step {self.step_number + 1} of {self.max_steps} max possible steps\n'
        time_str = datetime.now().isoformat()
        step_info_description += f'Current date and time: {time_str}'
        
        available_actions_description = [tcp.to_string(tool) for tool in tcp.list()]
        available_actions_description = "\n".join(available_actions_description)
        
        todo_contents = await self._get_todo_contents()
        
        return {
            "task": task,
            "step_info": step_info_description,
            "available_actions": available_actions_description,
            "todo_contents": todo_contents,
        }
        
    async def _get_environment_state(self) -> Dict[str, Any]:
        """Get the environment state."""
        environment_state = ""
        for env_name in ecp.list():
            env_state = await ecp.get_state(env_name)
            state_string = env_state.state
            environment_state += dedent(f"""
                <{env_name}_state>
                {state_string}
                </{env_name}_state>
            """)
        
        return {
            "environment_state": environment_state,
        }
        
    async def _get_messages(self, task: str) -> List[BaseMessage]:
        
        system_input_variables = {}
        environment_rules = ""
        for env_name in ecp.list():
            environment_rules += f"{ecp.get_info(env_name).rules}\n"
        system_input_variables.update(dict(
            environment_rules=environment_rules,
        ))
        system_message = self.prompt_manager.get_system_message(system_input_variables)
        
        agent_input_variables = {}
        agent_history = await self._get_agent_history()
        agent_state = await self._get_agent_state(task)
        environment_state = await self._get_environment_state()
        
        agent_input_variables.update(agent_history)
        agent_input_variables.update(agent_state)
        agent_input_variables.update(environment_state)
        
        agent_message = self.prompt_manager.get_agent_message(agent_input_variables)
        
        messages = [
            system_message,
            agent_message,
        ]
        
        return messages
    
    async def _get_cua_messages(self, proposed_action: Any) -> Dict[str, Any]:
        """Correct the action."""
        system_input_variables = {}
        system_message = self.cua_prompt_manager.get_system_message(system_input_variables)
        
        agent_input_variables = {}
        environment_state = await self._get_environment_state()
        agent_input_variables.update(environment_state)
        
        thinking = proposed_action.thinking
        evaluation_previous_goal = proposed_action.evaluation_previous_goal
        memory = proposed_action.memory
        next_goal = proposed_action.next_goal
        action = proposed_action.action
        
        proposed_action_string = dedent(f"""
            - Thinking: {thinking}
            - Evaluation of Previous Goal: {evaluation_previous_goal}
            - Memory: {memory}
            - Next Goal: {next_goal}
            - Action: {action}
        """)
        agent_input_variables["proposed_action"] = proposed_action_string
        
        agent_message = self.cua_prompt_manager.get_agent_message(agent_input_variables)

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
            
            action = actions[0] # First action
            action = action.model_dump() # Convert to dict
            
            logger.info(f"| 💭 Thinking: {thinking[:self.log_max_length]}...")
            logger.info(f"| 🔧 Evaluation of Previous Goal: {evaluation_previous_goal}")
            logger.info(f"| 🔧 Memory: {memory}")
            logger.info(f"| 🎯 Next Goal: {next_goal}")
            logger.info(f"| 📝 Action: {action}")
            
            extra_thinking = ""
            corrected_action = {}
            if self.if_correct_action:
            # Correct the action
                try:
                    messages = await self._get_cua_messages(think_output)

                    cua_output = await self.cua_model.ainvoke(messages, reasoning={"summary": "concise"})
                    
                    contents = cua_output.content
                    for content in contents:
                        if content['type'] == 'text':
                            extra_thinking += content["text"]
                        elif content['type'] == 'reasoning':
                            summary = " ".join([item['text'] for item in content["summary"]])
                            extra_thinking += summary
                        elif content['type'] == 'computer_call':
                            action_dict = content['action']
                            corrected_action['name'] = action_dict['type']
                            corrected_action['args'] = {
                                key: value for key, value in action_dict.items() if key != 'type'
                            }
                    
                    logger.info(f"| 📝 Corrected Action: {corrected_action}")
                    
                except Exception as e:
                    logger.error(f"| Error in correcting action: {e}")
            
            if extra_thinking:
                thinking += extra_thinking
            if corrected_action:
                action.update(corrected_action)
                
            # Get tool name and args
            tool_name = action['name']
            tool_args = action['args']
                
            # Execute the first action
            action_results = []
            
            logger.info(f"| 📝 Action Name: {tool_name}, Args: {tool_args}")
            
            tool_result = await tcp.ainvoke(tool_name, input=tool_args)
            if isinstance(tool_result, ToolResponse):
                tool_result = tool_result.content
            else:
                tool_result = str(tool_result)
            
            logger.info(f"| ✅ Action completed successfully")
            logger.info(f"| 📄 Results: {str(tool_result)}...")
            
            # Update action with result
            action["output"] = tool_result
            action_results.append(action)
                
            if tool_name == "done":
                done = True
                final_result = tool_result
            
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