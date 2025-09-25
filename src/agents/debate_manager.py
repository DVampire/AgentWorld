"""Multi-Agent Debate Manager for coordinating multiple agents in a debate."""

import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator, Generator, Type
from datetime import datetime
import random
from pydantic import BaseModel, Field, ConfigDict

from src.logger import logger
from src.agents.protocol.agent import BaseAgent
from src.agents.protocol import acp
from src.infrastructures.models import model_manager
from src.agents.prompts import prompt_manager


class DebateManagerInputArgs(BaseModel):
    topic: str = Field(description="The debate topic.")
    initial_message: Optional[str] = Field(default=None, description="Optional initial message to start the debate.")
    max_rounds: Optional[int] = Field(default=10, description="Maximum number of debate rounds.")


@acp.agent()
class DebateManagerAgent(BaseAgent):
    """Manages multi-agent debate sessions."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="debate_manager", description="The name of the debate manager.")
    type: str = Field(default="Agent", description="The type of the debate manager.")
    description: str = Field(default="A debate manager that coordinates multiple agents in a debate.", description="The description of the debate manager.")
    args_schema: Type[DebateManagerInputArgs] = Field(default=DebateManagerInputArgs, description="The args schema of the debate manager.")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the debate manager.")
    
    def __init__(
        self,
        name: str,
        description: str,
        workdir: str,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        max_steps: int = 1,
        review_steps: int = 1,
        log_max_length: int = 1000,
        max_rounds: int = 10,
        **kwargs
    ):
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
            max_rounds=max_rounds,
            **kwargs)
        
        self.name = name
        self.description = description
        
        # No tools needed for simple chat
        self.tools = []
        self.model = self.model  # Use the base model without tool binding
        
    async def start_debate(self, 
                           topic: str, 
                           files: Optional[List[str]] = None,
                           agents: List[str] = [],
                           initial_message: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """Start a multi-agent debate on the given topic."""
        logger.info(f"| 🎯 Starting debate on topic: {topic}")
        
        # Get available agents from acp
        available_agents = agents
        logger.info(f"| 🤖 Available agents: {available_agents}")
        
        # Initialize debate
        self.global_conversation_history = []
        self.current_round = 0
        self.active_agents = available_agents.copy()
        
        # Add initial topic message
        if initial_message:
            self.global_conversation_history.append({
                "agent": "moderator",
                "type": "topic",
                "content": f"Debate topic: {topic}\nInitial message: {initial_message}",
                "timestamp": datetime.now().isoformat()
            })
        else:
            self.global_conversation_history.append({
                "agent": "moderator", 
                "type": "topic",
                "content": f"Debate topic: {topic}",
                "timestamp": datetime.now().isoformat()
            })
        
        yield {
            "type": "debate_start",
            "topic": topic,
            "agents": available_agents,
            "content": f"Debate started on: {topic}"
        }
        
        # Start debate rounds
        current_message = initial_message or topic
        
        while self.current_round < self.max_rounds and self.active_agents:
            self.current_round += 1
            logger.info(f"| 🔄 Debate Round {self.current_round}/{self.max_rounds}")
            
            # Select next agent to speak (round-robin)
            speaking_agent_name = self._select_speaking_agent()
            if not speaking_agent_name:
                logger.info("| 🏁 No more agents want to speak")
                break
            
            # Let the agent speak through acp
            async for event in self._agent_speak(speaking_agent_name, current_message):
                yield event
                
                # Update global history
                if event["type"] in ["response_complete", "decision"]:
                    self.global_conversation_history.append({
                        "agent": event["agent"],
                        "type": event["type"],
                        "content": event["content"],
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Check if debate should continue
            if not self._should_continue_debate():
                logger.info("| 🏁 Debate reached natural conclusion.")
                break
                
            # Prepare next message (could be a follow-up question)
            current_message = await self._generate_next_topic()
        
        # End debate
        yield {
            "type": "debate_end",
            "rounds": self.current_round,
            "content": f"Debate completed after {self.current_round} rounds"
        }
    
    def _select_speaking_agent(self) -> Optional[str]:
        """Select which agent should speak next."""
        if not self.active_agents:
            return None
        try:
            return self.active_agents[self.current_round % len(self.active_agents)]
        except (IndexError, ZeroDivisionError):
            return None
    
    async def _agent_speak(self, agent_name: str, message: str) -> AsyncGenerator[Dict, None]:
        """Let an agent speak in the debate."""
        logger.info(f"| 🎤 {agent_name} is speaking...")
        
        try:
            # Call agent through acp
            result = await acp.ainvoke(
                name=agent_name,
                input={
                    "task": message,
                    "files": []
                }
            )
            
            # Process the result
            if result:
                yield {
                    "agent": agent_name,
                    "type": "response_complete",
                    "content": str(result),
                    "should_continue": True
                }
            else:
                yield {
                    "agent": agent_name,
                    "type": "decision",
                    "content": f"{agent_name} decided not to respond",
                    "should_continue": False
                }
                
                # Remove from active agents
                if agent_name in self.active_agents:
                    self.active_agents.remove(agent_name)
                    logger.info(f"| 🚫 {agent_name} has left the debate")
                    yield {
                        "type": "agent_exit",
                        "agent": agent_name,
                        "content": f"{agent_name} has decided to leave the debate"
                    }
                    
        except Exception as e:
            logger.error(f"| ❌ Error with {agent_name}: {e}")
            yield {
                "type": "agent_error",
                "agent": agent_name,
                "content": f"Error with {agent_name}: {str(e)}"
            }
    
    def _should_continue_debate(self) -> bool:
        """Determine if the debate should continue."""
        # Simple logic: continue if we have active agents and haven't reached max rounds
        return len(self.active_agents) > 0 and self.current_round < self.max_rounds
    
    async def _generate_next_topic(self) -> str:
        """Generate the next topic or question for the debate."""
        # This could be more sophisticated - analyze the conversation
        # and generate relevant follow-up questions
        
        follow_up_questions = [
            "What are your thoughts on this point?",
            "Can you elaborate on that?",
            "What evidence supports your position?",
            "How do you respond to the previous argument?",
            "What are the implications of this?",
            "Can you provide a counter-argument?"
        ]
        
        return random.choice(follow_up_questions)
    
    async def ainvoke(self,
                      topic: str, 
                      files: Optional[List[str]] = None, 
                      agents: List[str] = []):
        """Main entry point for debate manager through acp."""
        logger.info(f"| 🎯 DebateManagerAgent starting debate: {topic[:100]}...")
        
        # Start debate and collect all events
        events = []
        debate_output = []
        
        async for event in self.start_debate(topic, files, agents):
            events.append(event)
            
            # Collect meaningful output for the user
            if event["type"] == "debate_start":
                debate_output.append(f"🎯 {event['content']}")
                debate_output.append(f"🤖 Participants: {', '.join(event['agents'])}")
                debate_output.append("=" * 60)
                
            elif event["type"] == "response_complete":
                agent_name = event.get("agent", "Unknown")
                content = event.get("content", "")
                debate_output.append(f"\n🤖 {agent_name}: {content}")
                debate_output.append("-" * 40)
                
            elif event["type"] == "decision":
                agent_name = event.get("agent", "Unknown")
                content = event.get("content", "")
                debate_output.append(f"\n🤔 {agent_name} decision: {content}")
                
            elif event["type"] == "agent_exit":
                agent_name = event.get("agent", "Unknown")
                debate_output.append(f"\n🚪 {agent_name} has left the debate")
                
            elif event["type"] == "debate_end":
                rounds = event.get("rounds", 0)
                debate_output.append(f"\n🏁 Debate ended after {rounds} rounds")
                debate_output.append("=" * 60)
        
        # Return formatted debate output
        result = "\n".join(debate_output)
        logger.info(f"| ✅ Debate completed with {len(events)} events")
        
        return result
    
    def get_debate_summary(self) -> Dict[str, Any]:
        """Get a summary of the debate."""
        return {
            "total_rounds": self.current_round,
            "active_agents": self.active_agents,
            "conversation_history": self.global_conversation_history,
            "debate_complete": len(self.active_agents) == 0 or self.current_round >= 10
        }
