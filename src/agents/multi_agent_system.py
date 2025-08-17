"""Multi-agent system for coordinating multiple agents."""

from typing import Dict, List, Any, Optional, Callable
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import json
import asyncio

from .base_agent import BaseAgent
from .simple_agent import SimpleAgent


class MultiAgentSystem:
    """Multi-agent system that coordinates multiple agents."""
    
    def __init__(self, name: str = "multi_agent_system"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_graph = None
        self.routing_function = None
        self.state = {}
    
    def add_agent(self, agent: BaseAgent):
        """Add an agent to the system."""
        self.agents[agent.name] = agent
    
    def remove_agent(self, agent_name: str):
        """Remove an agent from the system."""
        if agent_name in self.agents:
            del self.agents[agent_name]
    
    def set_routing_function(self, routing_function: Callable):
        """Set the routing function to determine which agent should handle a message."""
        self.routing_function = routing_function
    
    async def create_workflow(self):
        """Create the workflow graph for the multi-agent system."""
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes for each agent
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, self._create_agent_node(agent))
        
        # Add routing node
        workflow.add_node("router", self._router_node)
        
        # Add conditional edges from router to agents
        agent_edges = {name: name for name in self.agents.keys()}
        agent_edges[END] = END
        workflow.add_conditional_edges(
            "router",
            self._route_to_agent,
            agent_edges
        )
        
        # Add edges from agents back to router
        for agent_name in self.agents.keys():
            workflow.add_conditional_edges(
                agent_name,
                self._should_continue,
                {
                    "router": "router",
                    END: END
                }
            )
        
        # Set entry point
        workflow.set_entry_point("router")
        
        self.workflow_graph = workflow.compile()
        return self.workflow_graph
    
    def _create_agent_node(self, agent: BaseAgent):
        """Create a node for a specific agent."""
        async def agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            messages = state.get("messages", [])
            current_message = messages[-1] if messages else None
            
            if current_message:
                # Process the message with the agent
                result = await agent.process_message(current_message.content, {"messages": messages})
                return result
            
            return state
        
        return agent_node
    
    async def _router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Router node that determines which agent should handle the message."""
        messages = state.get("messages", [])
        current_message = messages[-1] if messages else None
        
        if current_message:
            # Use routing function to determine next agent
            if self.routing_function:
                next_agent = self.routing_function(current_message.content, state)
                state["next_agent"] = next_agent
            else:
                # Default routing: use the first agent
                state["next_agent"] = list(self.agents.keys())[0]
        
        return state
    
    def _route_to_agent(self, state: Dict[str, Any]) -> str:
        """Route to the appropriate agent."""
        next_agent = state.get("next_agent")
        if next_agent in self.agents:
            return next_agent
        return END
    
    def _should_continue(self, state: Dict[str, Any]) -> str:
        """Determine if the conversation should continue."""
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        if last_message and isinstance(last_message, AIMessage):
            content = last_message.content.lower()
            if any(phrase in content for phrase in ["goodbye", "bye", "end", "finished", "complete"]):
                return END
        
        return "router"
    
    async def process_message(self, message: str, initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a message through the multi-agent system."""
        if not self.workflow_graph:
            await self.create_workflow()
        
        # Initialize state
        state = initial_state or {}
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=message))
        state["messages"] = messages
        
        # Run the workflow
        result = await self.workflow_graph.ainvoke(state)
        return result
    
    async def process_messages_concurrently(self, messages: List[str], initial_state: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Process multiple messages concurrently."""
        tasks = []
        for message in messages:
            task = self.process_message(message, initial_state)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """List all agent names."""
        return list(self.agents.keys())
    
    def get_conversation_history(self) -> List:
        """Get the conversation history."""
        return self.state.get("messages", [])


class MultiAgentState:
    """State class for multi-agent workflow."""
    
    def __init__(self, messages: List = None, next_agent: str = None):
        self.messages = messages or []
        self.next_agent = next_agent
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)


# Predefined routing functions
def round_robin_routing(message: str, state: Dict[str, Any]) -> str:
    """Round-robin routing function."""
    agents = list(state.get("agents", {}).keys())
    if not agents:
        return None
    
    # Get the last agent that was used
    last_agent = state.get("last_agent")
    if last_agent in agents:
        current_index = agents.index(last_agent)
        next_index = (current_index + 1) % len(agents)
    else:
        next_index = 0
    
    next_agent = agents[next_index]
    state["last_agent"] = next_agent
    return next_agent


def keyword_based_routing(message: str, state: Dict[str, Any]) -> str:
    """Keyword-based routing function."""
    agents = state.get("agents", {})
    
    # Define keywords for each agent
    agent_keywords = {
        "researcher": ["research", "study", "analysis", "investigate", "find"],
        "writer": ["write", "compose", "draft", "create", "generate"],
        "coder": ["code", "program", "develop", "implement", "debug"],
        "planner": ["plan", "organize", "schedule", "coordinate", "manage"]
    }
    
    message_lower = message.lower()
    
    # Find the agent with the most matching keywords
    best_agent = None
    max_matches = 0
    
    for agent_name, keywords in agent_keywords.items():
        if agent_name in agents:
            matches = sum(1 for keyword in keywords if keyword in message_lower)
            if matches > max_matches:
                max_matches = matches
                best_agent = agent_name
    
    return best_agent or list(agents.keys())[0] if agents else None


def llm_based_routing(llm: BaseLanguageModel, agents: Dict[str, BaseAgent]) -> Callable:
    """Create an LLM-based routing function."""
    async def routing_function(message: str, state: Dict[str, Any]) -> str:
        agent_names = list(agents.keys())
        agent_descriptions = [f"{name}: {agents[name].__class__.__name__}" for name in agent_names]
        
        prompt = f"""Given the following message and available agents, determine which agent should handle it:

Message: {message}

Available agents:
{chr(10).join(agent_descriptions)}

Respond with only the agent name:"""

        response = await llm.ainvoke(prompt)
        selected_agent = response.content.strip()
        
        return selected_agent if selected_agent in agent_names else agent_names[0]
    
    return routing_function
