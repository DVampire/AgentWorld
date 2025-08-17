"""Simple agent implementation using LangGraph."""

from typing import Dict, List, Any, Optional, Union
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
import asyncio

from .base_agent import BaseAgent


class SimpleAgent(BaseAgent):
    """Simple agent implementation with tool usage capabilities."""
    
    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        llm: Optional[BaseLanguageModel] = None,
        prompt_template: Optional[str] = None,
        prompt_name: Optional[str] = None,
        tools: Optional[List[Union[str, BaseTool]]] = None,
        mcp_tools: Optional[List[Dict]] = None,
        **kwargs
    ):
        super().__init__(name, model_name, llm, prompt_template, prompt_name, tools, mcp_tools, **kwargs)
        self.agent_executor = None
        self._setup_agent_executor()
    
    def _setup_agent_executor(self):
        """Setup the agent executor with tools."""
        if not self.tools:
            # If no tools, create a simple agent without tools
            self.agent_executor = None
            return
        
        # Create agent with tools
        agent = create_openai_functions_agent(
            llm=self.model,
            tools=self.tools,
            prompt=self.prompt_template
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def create_graph(self) -> StateGraph:
        """Create the agent's state graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        
        if self.tools:
            # Add tool node if tools are available
            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)
            
            # Add edges
            workflow.add_edge("agent", "tools")
            workflow.add_conditional_edges(
                "tools",
                self._should_continue,
                {
                    "agent": "agent",
                    END: END
                }
            )
        else:
            # Simple flow without tools
            workflow.add_edge("agent", END)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        return workflow.compile()
    
    async def _agent_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent node that processes messages."""
        messages = state.get("messages", [])
        
        if self.agent_executor:
            # Use agent executor with tools
            result = await self.agent_executor.ainvoke({
                "input": messages[-1].content if messages else "",
                "chat_history": messages[:-1] if len(messages) > 1 else []
            })
            
            # Add the response to messages
            response_message = AIMessage(content=result["output"])
            messages.append(response_message)
            
            return {"messages": messages}
        else:
            # Simple LLM call without tools
            prompt_vars = self.get_prompt_variables()
            prompt_vars.update({
                "input": messages[-1].content if messages else "",
                "chat_history": "\n".join([msg.content for msg in messages[:-1]]) if len(messages) > 1 else ""
            })
            
            response = await self.model.ainvoke(self.prompt_template.format(**prompt_vars))
            response_message = AIMessage(content=response.content)
            messages.append(response_message)
            
            return {"messages": messages}
    
    def _should_continue(self, state: Dict[str, Any]) -> str:
        """Determine if the agent should continue or end."""
        # Check if the last message indicates the conversation should end
        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        
        if last_message and isinstance(last_message, AIMessage):
            content = last_message.content.lower()
            if any(phrase in content for phrase in ["goodbye", "bye", "end", "finished", "complete"]):
                return END
        
        return "agent"
    
    async def process_message(self, message: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message and return updated state."""
        if not self.state_graph:
            self.state_graph = await self.create_graph()
        
        # Add the new message to state
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=message))
        
        # Run the graph
        result = await self.state_graph.ainvoke({"messages": messages})
        return result
    
    def change_prompt_template(self, prompt_name: str):
        """Change the agent's prompt template and recreate the executor."""
        self.set_prompt_template(prompt_name)
        self._setup_agent_executor()  # Recreate executor with new prompt
    
    def change_model(self, model_name: str):
        """Change the agent's model and recreate the executor."""
        self.set_model(model_name)
        self._setup_agent_executor()  # Recreate executor with new model
    
    def get_prompt_info(self) -> Dict[str, Any]:
        """Get information about the current prompt template."""
        template_config = self.prompt_manager.get_template_config("default")
        if template_config:
            return {
                "name": "default",
                "description": template_config.get("description", "No description"),
                "input_variables": self.prompt_template.input_variables,
                "agent_type": template_config.get("agent_type", "general"),
                "specialization": template_config.get("specialization", "general")
            }
        return {
            "name": "custom",
            "description": "Custom prompt template",
            "input_variables": self.prompt_template.input_variables,
            "agent_type": "custom",
            "specialization": "custom"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        # Find which model is currently being used
        for model_name in self.model_manager.list_models():
            model = self.model_manager.get_model(model_name)
            if model and model.llm == self.model:
                return self.model_manager.get_model_info(model_name)
        
        return {
            "name": "unknown",
            "type": "unknown",
            "description": "Model information not available"
        }


class AgentState:
    """State class for agent workflow."""
    
    def __init__(self, messages: List = None):
        self.messages = messages or []
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def get(self, key, default=None):
        return getattr(self, key, default)
