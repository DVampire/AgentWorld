"""Simple chat agent for human conversation."""
from typing import List, Optional, Dict, Any, Type
from langchain_core.messages import BaseMessage, HumanMessage
from datetime import datetime
import asyncio
from pydantic import BaseModel, Field, ConfigDict
import json

from src.agent.types import Agent, ThinkOutputBuilder, InputArgs
from src.logger import logger
from src.agent.server import acp
from src.memory import memory_manager, SessionInfo, EventType
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT

@AGENT.register_module(force=True)
class SimpleChatAgent(Agent):
    """Simple chat agent for human conversation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="simple_chat", description="The name of the simple chat agent.")
    description: str = Field(default="A simple chat agent that can have conversations with humans.", description="The description of the simple chat agent.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the simple chat agent.")
    
    def __init__(
        self,
        name: str,
        description: str,
        workdir: str,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        max_steps: int = 1,  # Simple chat only needs one step
        review_steps: int = 1,
        log_max_length: int = 1000,
        **kwargs
    ):
        # Set default prompt name for simple chat
        if not prompt_name:
            prompt_name = "simple_chat"
        
        super().__init__(
            name=name,
            description=description,
            workdir=workdir,
            model_name=model_name,
            prompt_name=prompt_name,
            max_steps=max_steps,
            review_steps=review_steps,
            log_max_length=log_max_length,
            **kwargs)
        
        self.name = name
        self.description = description
        
        # No tools needed for simple chat
        self.tools = []
        self.model = self.model  # Use the base model without tool binding
    
    async def _generate_session_info(self, message: str) -> SessionInfo:
        """Generate session info for the chat."""
        prompt = f"""Generate a session info for a simple chat agent.
        
        The user message is: {message}
        
        Create a concise session ID and description for this conversation."""
        
        messages = [HumanMessage(content=prompt)]
        
        model_response = await model_manager(
            model=self.model_name,
            messages=messages,
            response_format=SessionInfo
        )
        result = model_response.extra["parsed_model"]
        
        timestamp = datetime.now().isoformat()
        session_id = f"{self.name}_{timestamp}"
        
        return SessionInfo(session_id=session_id, description=result.description)
    
    async def _get_agent_history(self) -> str:
        """Get the agent conversation history."""
        state = await memory_manager.get_state(memory_name=self.memory_name, n=self.review_steps)
        
        events = state["events"]
        conversation_history = ""
        
        for event in events:
            if event.event_type == EventType.TASK_START:
                conversation_history += f"User: {event.data['message']}\n"
            elif event.event_type == EventType.ACTION_STEP:
                conversation_history += f"Assistant: {event.data['response']}\n"
        
        return conversation_history
    
    async def _get_messages(self, message: str) -> List[BaseMessage]:
        """Generate messages for the conversation."""
        system_modules = self.prompt_modules.copy()
        # Infer prompt name from agent's prompt_name
        if self.prompt_name:
            system_prompt_name = f"{self.prompt_name}_system_prompt"
            agent_message_prompt_name = f"{self.prompt_name}_agent_message_prompt"
        else:
            system_prompt_name = "simple_chat_system_prompt"
            agent_message_prompt_name = "simple_chat_agent_message_prompt"
        
        system_message = await prompt_manager.get_system_message(
            prompt_name=system_prompt_name,
            modules=system_modules, 
            reload=False
        )
        
        # Use global conversation history if available
        conversation_history = ""
        if hasattr(self, '_current_global_history') and self._current_global_history:
            conversation_history = self._format_global_history(self._current_global_history)
        
        agent_message_modules = self.prompt_modules.copy()
        agent_message_modules.update({
            "user_message": message,
            "conversation_history": conversation_history,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
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
    
    async def _should_continue_conversation(self, current_message: str, conversation_history: str = "", last_response: str = "") -> tuple[bool, str]:
        """Let LLM decide whether the conversation should continue based on current message and context."""
        decision_prompt = f"""
You are a helpful AI assistant participating in a debate or discussion. 

Current message: "{current_message}"
{conversation_history}
{f"Last response: {last_response}" if last_response else ""}

Please decide whether you should respond to this message. In a debate context, you should be MORE LIKELY to respond to:
- Questions or statements about topics
- Arguments or counter-arguments
- Requests for opinions or analysis
- Discussion points
- Any substantive content

Only decline to respond if:
- The message is completely inappropriate or harmful
- The message is pure spam or gibberish
- The message is completely unrelated to any discussion topic

Respond with a JSON format:
{{
    "should_continue": true/false,
    "reasoning": "Brief explanation of your decision",
    "response_type": "helpful/decline/redirect/end"
}}

Remember: In debates, it's better to engage and provide your perspective rather than decline to respond.

"""

        messages = [HumanMessage(content=decision_prompt)]
        model_response = await model_manager(model=self.model_name, messages=messages)
        response_text = model_response.message
        
        try:
            decision = json.loads(response_text)
            should_continue = decision.get("should_continue", True)
            reasoning = decision.get("reasoning", "No reasoning provided")
            response_type = decision.get("response_type", "helpful")
            return should_continue, f"{reasoning} (Type: {response_type})"
        except:
            # If JSON parsing fails, default to continuing
            return True, "Could not parse decision, defaulting to continue"

    async def _generate_proactive_question(self, last_response: str, conversation_history: str) -> str:
        """Generate a proactive question or topic to continue the conversation."""
        proactive_prompt = f"""
You are an AI assistant that wants to continue a meaningful conversation. 

Your last response: "{last_response}"

{conversation_history}

Generate a natural follow-up question or topic that you would ask to continue the conversation. This should be:
- A thoughtful follow-up question related to the current topic
- A deeper exploration of the subject matter
- A related but interesting tangent
- A question that shows genuine curiosity and engagement

The question should:
- Be natural and conversational
- Show interest in learning more
- Be relevant to the current discussion
- Encourage further exploration of the topic

Keep it engaging but not too complex. Make it sound like you're genuinely curious.
"""

        messages = [HumanMessage(content=proactive_prompt)]
        model_response = await model_manager(model=self.model_name, messages=messages)
        return model_response.message

    async def _get_user_input(self, question: str) -> Optional[str]:
        """Get real user input with the given question prompt."""
        try:
            # Get real user input in an async way
            loop = asyncio.get_event_loop()
            user_input = await loop.run_in_executor(None, input)
            user_input = user_input.strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'end']:
                return None
            elif user_input == "":
                return None
            else:
                return user_input
        except (EOFError, KeyboardInterrupt):
            return None
        except Exception as e:
            logger.error(f"Error getting user input: {e}")
            return None

    async def __call__(self, task: str, files: Optional[List[str]] = None, global_conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Main entry point for simple chat agent through acp.
        
        Args:
            task (str): The conversation task/message to process.
            files (Optional[List[str]]): The files to attach to the task.
            global_conversation_history (Optional[List[Dict]]): The global conversation history for multi-agent scenarios.
            
        Returns:
            str: The final conversation result as a string.
        """
        logger.info(f"| 💬 SimpleChatAgent starting multi-turn conversation: {task[:self.log_max_length]}...")
        
        # Generate session info
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
        
        # Initialize conversation
        task_id = "chat_" + datetime.now().strftime("%Y%m%d-%H%M%S")
        current_message = task
        conversation_round = 0
        max_rounds = 10  # Prevent infinite loops
        
        logger.info(f"| 🚀 Starting conversation session: {session_id}")
        
        while conversation_round < max_rounds:
            conversation_round += 1
            logger.info(f"| 🔄 Conversation round {conversation_round}/{max_rounds}")
            
            # Get conversation history for decision making
            conversation_history = await self._get_agent_history()
            
            # Let LLM decide whether to continue the conversation
            should_continue, reasoning = await self._should_continue_conversation(current_message, conversation_history)
            logger.info(f"| 🤔 Decision: {reasoning}")
            
            if not should_continue:
                logger.info(f"| 🚫 Agent decided not to continue (round {conversation_round})")
                break
            
            # Add user message event
            await memory_manager.add_event(
                memory_name=self.memory_name,
                step_number=self.step_number, 
                event_type="task_start", 
                data=dict(message=current_message),
                agent_name=self.name,
                task_id=task_id
            )
            
            # Generate response
            messages = await self._get_messages(current_message)
            model_response = await model_manager(model=self.model_name, messages=messages)
            response_text = model_response.message
            
            logger.info(f"| 🤖 Assistant response: {response_text[:self.log_max_length]}...")
            
            # Add response event
            await memory_manager.add_event(
                memory_name=self.memory_name,
                step_number=self.step_number,
                event_type="action_step",
                data=dict(response=response_text),
                agent_name=self.name,
                task_id=task_id
            )
            
            # Check if conversation should continue after response
            should_continue_after, continue_reasoning = await self._should_continue_conversation("", conversation_history, response_text)
            logger.info(f"| 🤔 Continue after response: {continue_reasoning}")
            
            if not should_continue_after:
                logger.info(f"| 🏁 Agent decided to end conversation after response (round {conversation_round})")
                break
            
            # Agent proactively generates next question and waits for user input
            next_question = await self._generate_proactive_question(response_text, conversation_history)
            logger.info(f"| 🤔 Agent's next question: {next_question[:self.log_max_length]}...")
            
            # Wait for user input with timeout
            try:
                logger.info(f"| 🔄 Waiting for user input...")
                user_input = await asyncio.wait_for(
                    self._get_user_input(next_question), 
                    timeout=30.0  # 30 seconds timeout
                )
                if user_input is None:
                    logger.info(f"| ⏰ Timeout waiting for user input, ending conversation")
                    break
                current_message = user_input
                logger.info(f"| 👤 User response: {current_message[:self.log_max_length]}...")
            except asyncio.TimeoutError:
                logger.info(f"| ⏰ Timeout waiting for user input, ending conversation")
                break
        
        # Add task end event
        await memory_manager.add_event(
            memory_name=self.memory_name,
            step_number=self.step_number,
            event_type="task_end",
            data=dict(result=f"Conversation completed after {conversation_round} rounds"),
            agent_name=self.name,
            task_id=task_id
        )
        
        # End session
        await memory_manager.end_session(memory_name=self.memory_name, session_id=session_id)
        
        logger.info(f"| ✅ Multi-turn conversation completed after {conversation_round} rounds")
        
        return f"Conversation completed in {conversation_round} rounds"

    def _format_global_history(self, global_history: List[Dict]) -> str:
        """Format global conversation history for the agent."""
        history_text = ""
        for entry in global_history:
            agent_name = entry.get("agent", "Unknown")
            content = entry.get("content", "")
            entry_type = entry.get("type", "message")
            
            if entry_type == "response_complete":
                history_text += f"{agent_name}: {content}\n"
            elif entry_type == "decision":
                history_text += f"[{agent_name} decision]: {content}\n"
        
        return history_text

    async def ainvoke_stream(self, task: str, files: Optional[List[str]] = None, global_conversation_history: Optional[List[Dict]] = None):
        """Process conversation with streaming output for multi-agent debate."""
        logger.info(f"| 💬 {self.name} starting debate turn: {task[:self.log_max_length]}...")
        
        # Use global conversation history if provided
        if global_conversation_history:
            conversation_history = self._format_global_history(global_conversation_history)
        else:
            conversation_history = await self._get_agent_history()
        
        # Let LLM decide whether to respond
        should_continue, reasoning = await self._should_continue_conversation(task, conversation_history)
        logger.info(f"| 🤔 {self.name} decision: {reasoning}")
        
        if not should_continue:
            logger.info(f"| 🚫 {self.name} decided not to continue")
            yield {
                "agent": self.name,
                "type": "decision",
                "content": f"Decided not to respond: {reasoning}",
                "should_continue": False
            }
            return
        
        # Generate response
        messages = await self._get_messages(task)
        
        # Generate response (using regular invoke for now, can be enhanced with streaming later)
        try:
            model_response = await model_manager(model=self.model_name, messages=messages)
            response_content = model_response.message
            
            # Simulate streaming by yielding the full response
            yield {
                "agent": self.name,
                "type": "response_chunk",
                "content": response_content,
                "partial_response": response_content
            }
            
        except Exception as e:
            logger.error(f"Error generating response for {self.name}: {e}")
            yield {
                "agent": self.name,
                "type": "error",
                "content": f"Error generating response: {str(e)}",
                "should_continue": False
            }
            return
        
        logger.info(f"| 🤖 {self.name} response: {response_content[:self.log_max_length]}...")
        
        # Final response
        yield {
            "agent": self.name,
            "type": "response_complete",
            "content": response_content,
            "should_continue": True
        }

    async def ainvoke_simple(self, task: str, files: Optional[List[str]] = None, global_conversation_history: Optional[List[Dict]] = None):
        """Simple single response for debate scenarios."""
        logger.info(f"| 💬 {self.name} responding to: {task[:self.log_max_length]}...")
        
        # Set global conversation history for this agent
        self._current_global_history = global_conversation_history or []
        
        # Get conversation history for decision making
        if global_conversation_history:
            conversation_history = self._format_global_history(global_conversation_history)
        else:
            conversation_history = ""
        
        # Let LLM decide whether to respond
        should_continue, reasoning = await self._should_continue_conversation(task, conversation_history)
        logger.info(f"| 🤔 {self.name} decision: {reasoning}")
        
        if not should_continue:
            logger.info(f"| 🚫 {self.name} decided not to respond")
            return None
        
        try:
            # Generate response
            logger.info(f"| 🔧 {self.name} getting messages...")
            messages = await self._get_messages(task)
            logger.info(f"| 🔧 {self.name} got {len(messages)} messages")
            
            logger.info(f"| 🔧 {self.name} calling model...")
            model_response = await model_manager(model=self.model_name, messages=messages)
            logger.info(f"| 🔧 {self.name} got response: {type(model_response)}")
            
            # Extract response content
            response_text = model_response.message
            
            logger.info(f"| 🤖 {self.name} response: {response_text[:self.log_max_length]}...")
            
            return response_text
            
        except Exception as e:
            logger.error(f"| ❌ Error in ainvoke_simple for {self.name}: {e}")
            import traceback
            logger.error(f"| ❌ Traceback: {traceback.format_exc()}")
            return None
