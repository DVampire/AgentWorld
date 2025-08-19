"""Tool calling agent implementation using LangChain's built-in agents with multi-turn conversation support."""

import json
import re
from typing import Dict, List, Any, Optional, Union, Callable, AsyncGenerator
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain.agents import AgentExecutor, create_openai_functions_agent
import asyncio

from src.agents.base_agent import BaseAgent
from src.registry import AGENTS
from src.logger import logger


@AGENTS.register_module(force=True)
class ToolCallingAgent(BaseAgent):
    """Tool calling agent implementation using LangChain's built-in agents with multi-turn conversation support."""
    
    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        llm: Optional[BaseLanguageModel] = None,
        prompt_template: Optional[str] = None,
        prompt_name: Optional[str] = None,
        tools: Optional[List[Union[str, BaseTool]]] = None,
        max_iterations: int = 10,
        verbose: bool = True,
        **kwargs
    ):
        # Set default prompt name for tool calling
        if not prompt_name:
            prompt_name = "tool_calling"
        
        super().__init__(name, model_name, llm, prompt_template, prompt_name, tools, **kwargs)
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.agent_executor = None
        self.conversation_history = []
        self._setup_agent_executor()
    
    def _setup_agent_executor(self):
        """Setup the agent executor with tools."""
        if not self.tools:
            # If no tools, create a simple agent without tools
            self.agent_executor = None
            logger.warning(f"⚠️ No tools available for agent {self.name}")
            return
        
        # Validate tools
        self._validate_tools()
        
        logger.info(f"🔧 Setting up agent executor with {len(self.tools)} tools")
        
        # Get tool names for the prompt
        tool_names = [tool.name for tool in self.tools]
        
        # Create a custom prompt that provides the required variables
        # For OpenAI functions agent, we need to format tools properly
        tools_description = "\n".join([f"- {tool.name}: {tool.description}" for tool in self.tools])
        
        # Create the prompt template with proper variable substitution
        custom_prompt = PromptTemplate(
            template=self.prompt_template.template,
            input_variables=["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
            partial_variables={
                "tools": tools_description,
                "tool_names": ", ".join(tool_names)
            }
        )
        
        # Create agent with tools using LangChain's built-in function
        self.agent = create_openai_functions_agent(
            llm=self.model,
            tools=self.tools,
            prompt=custom_prompt,
        )
        
        # Create agent executor with more aggressive settings
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=self.max_iterations,
            return_intermediate_steps=True,  # Return intermediate steps for debugging
            early_stopping_method="generate"  # Use generate method for early stopping
        )
        
        logger.info(f"✅ Agent executor created successfully with {len(self.tools)} tools")
        logger.info(f"📋 Available tools: {tool_names}")
    
    def _validate_tools(self):
        """Validate that tools are properly configured."""
        valid_tools = []
        for tool in self.tools:
            if hasattr(tool, 'name') and hasattr(tool, 'description'):
                valid_tools.append(tool)
                logger.debug(f"✅ Tool validated: {tool.name}")
            else:
                logger.warning(f"⚠️ Invalid tool found: {tool}")
        
        if len(valid_tools) != len(self.tools):
            logger.warning(f"⚠️ Some tools were invalid. Valid tools: {len(valid_tools)}/{len(self.tools)}")
            self.tools = valid_tools
    
    async def run(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Run a task using the agent's tools in standard mode.
        
        Args:
            task: The task to execute
            **kwargs: Additional parameters
            
        Returns:
            Dict containing the result and execution details
        """
        logger.info(f"🤖 {self.name} starting task: {task}")
        
        # Add the task to conversation history
        self.conversation_history.append(f"Human: {task}")
        
        # Get prompt variables
        prompt_vars = self.get_prompt_variables()
        prompt_vars.update({
            "chat_history": self._format_chat_history(),
            "input": task
        })
        
        # Standard mode - return final result
        return await self._run_standard(task, prompt_vars)
    
    async def run_streaming(self, task: str, **kwargs):
        """
        Run a task using the agent's tools in streaming mode.
        
        Args:
            task: The task to execute
            **kwargs: Additional parameters
            
        Yields:
            Dict containing streaming updates
        """
        logger.info(f"🤖 {self.name} starting streaming task: {task}")
        
        # Add the task to conversation history
        self.conversation_history.append(f"Human: {task}")
        
        # Get prompt variables
        prompt_vars = self.get_prompt_variables()
        prompt_vars.update({
            "chat_history": self._format_chat_history(),
            "input": task
        })
        
        # Streaming mode - yield step-by-step updates
        async for update in self._run_streaming(task, prompt_vars):
            yield update
    
    async def _run_standard(self, task: str, prompt_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Run task in standard mode and return final result."""
        try:
            result = await self.agent_executor.ainvoke(prompt_vars)
            
            # Check if tools were used
            intermediate_steps = result.get("intermediate_steps", [])
            tools_used = len(intermediate_steps) > 0
            
            # If no tools were used but tools are available, try to force tool usage
            if not tools_used and self.tools:
                logger.warning(f"⚠️ No tools were used for task: {task}")
                logger.info(f"🔄 Attempting to force tool usage...")
                
                # Try with more explicit prompt
                forced_result = await self._force_tool_usage(task, prompt_vars)
                if forced_result:
                    result = forced_result
                    intermediate_steps = result.get("intermediate_steps", [])
                    tools_used = len(intermediate_steps) > 0
            
            # Extract final answer
            final_response = await self._extract_final_answer(result["output"])
            self.conversation_history.append(f"Assistant: {final_response}")
            
            return {
                "task": task,
                "agent_name": self.name,
                "final_response": final_response,
                "conversation": self.conversation_history.copy(),
                "tool_results": intermediate_steps,
                "iterations": len(intermediate_steps),
                "max_iterations": self.max_iterations,
                "success": True,
                "executor_used": True,
                "tools_used": tools_used,
                "chat_history": self._format_chat_history(),
                "execution_mode": "agent_executor"
            }
            
        except Exception as e:
            logger.error(f"❌ Error in standard execution: {e}")
            error_response = f"Error: {str(e)}"
            self.conversation_history.append(f"Assistant: {error_response}")
            
            return {
                "task": task,
                "agent_name": self.name,
                "final_response": error_response,
                "conversation": self.conversation_history.copy(),
                "tool_results": [],
                "iterations": 1,
                "max_iterations": self.max_iterations,
                "success": False,
                "executor_used": True,
                "tools_used": False,
                "error": str(e),
                "chat_history": self._format_chat_history(),
                "execution_mode": "agent_executor"
            }
    
    async def _force_tool_usage(self, task: str, prompt_vars: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Force the agent to use tools by modifying the prompt."""
        try:
            # Create a more explicit prompt that forces tool usage
            forced_prompt_vars = prompt_vars.copy()
            forced_prompt_vars["input"] = f"IMPORTANT: You MUST use at least one tool to answer this question. {task}"
            
            # Add explicit tool usage instruction
            if "chat_history" in forced_prompt_vars:
                forced_prompt_vars["chat_history"] += "\n\nSystem: You must use tools to answer the next question."
            
            logger.info(f"🔄 Running with forced tool usage prompt")
            result = await self.agent_executor.ainvoke(forced_prompt_vars)
            
            # Check if tools were used this time
            intermediate_steps = result.get("intermediate_steps", [])
            if len(intermediate_steps) > 0:
                logger.info(f"✅ Successfully forced tool usage with {len(intermediate_steps)} steps")
                return result
            else:
                logger.warning(f"⚠️ Still no tools used even with forced prompt")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error in forced tool usage: {e}")
            return None
    
    async def _run_streaming(self, task: str, prompt_vars: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """Run task in streaming mode and yield step-by-step updates."""
        logger.info(f"🔍 Starting streaming execution")
        
        # Send task start
        yield {
            "type": "task_start",
            "task": task,
            "agent_name": self.name,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        try:
            # Use agent_executor.astream for streaming execution
            if not self.agent_executor:
                raise Exception("No agent executor available for streaming execution")
            
            # Stream the execution
            iteration_count = 0
            async for chunk in self.agent_executor.astream(prompt_vars):
                # Process each chunk based on actual astream output format
                if "actions" in chunk:
                    # Agent decided to use tools
                    iteration_count += 1
                    actions = chunk["actions"]
                    messages = chunk["messages"]
                    
                    # Extract tool call information from actions
                    for action in actions:
                        if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                            yield {
                                "type": "tool_calling",
                                "iteration": iteration_count,
                                "tool_name": action.tool,
                                "tool_input": action.tool_input,
                                "log": action.log,
                                "timestamp": asyncio.get_event_loop().time()
                            }
                
                elif "steps" in chunk:
                    # Tools executed and returned results
                    steps = chunk["steps"]
                    messages = chunk["messages"]
                    
                    # Process each step
                    for step in steps:
                        if hasattr(step, 'action') and hasattr(step, 'observation'):
                            yield {
                                "type": "tool_result",
                                "iteration": iteration_count,
                                "tool_name": step.action.tool,
                                "tool_input": step.action.tool_input,
                                "result": step.observation,
                                "timestamp": asyncio.get_event_loop().time()
                            }
                
                elif "output" in chunk:
                    # Final output received
                    final_response = await self._extract_final_answer(chunk["output"])
                    self.conversation_history.append(f"Assistant: {final_response}")
                    
                    # Send final response
                    yield {
                        "type": "final_response",
                        "final_response": final_response,
                        "total_iterations": iteration_count,
                        "timestamp": asyncio.get_event_loop().time()
                    }
            
        except Exception as e:
            logger.error(f"❌ Error in streaming execution: {e}")
            yield {
                "type": "error",
                "iteration": 1,
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
            yield {
                "type": "final_summary",
                "final_response": f"Error: {str(e)}",
                "total_iterations": 1,
                "timestamp": asyncio.get_event_loop().time()
            }
    
    def _should_stop(self, response: str) -> bool:
        """Check if we should stop execution."""
        # Check for explicit completion indicators
        completion_indicators = [
            "final answer:",
            "the answer is:",
            "result:",
            "conclusion:",
            "task completed",
            "task finished"
        ]
        
        response_lower = response.lower()
        for indicator in completion_indicators:
            if indicator in response_lower:
                return True
        
        return False
    
    def _is_task_complete(self, response: str) -> bool:
        """Check if the task appears to be complete."""
        # Check for completion indicators in the response
        completion_indicators = [
            "final answer:",
            "the answer is:",
            "result:",
            "conclusion:",
            "task completed",
            "task finished",
            "here's the answer",
            "the result is"
        ]
        
        response_lower = response.lower()
        for indicator in completion_indicators:
            if indicator in response_lower:
                return True
        
        return False
    
    async def _extract_final_answer(self, output: str) -> str:
        """Extract the final answer from a response using LLM."""
        
        # Use LLM to extract the final answer
        try:
            extract_prompt = f"""
Extract only the final answer or conclusion from this agent response. Ignore any intermediate reasoning, tool calls, or observations.

Agent Response:
{output}

Final Answer:"""

            response = await self.model.ainvoke(extract_prompt)
            return response.content.strip()
                
        except Exception as e:
            logger.warning(f"Failed to extract final answer with LLM: {e}")
            # Return the original output if LLM extraction fails
            return output.strip()
    
    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Find a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def _format_chat_history(self) -> str:
        """Format conversation history for the prompt."""
        if not self.conversation_history:
            return ""
        
        # Return the last few exchanges to keep context manageable
        # You can adjust this number based on your needs
        max_history = 10  # Keep last 10 exchanges
        recent_history = self.conversation_history[-max_history:]
        
        return "\n".join(recent_history)
    
    def add_message(self, message: str, is_human: bool = True):
        """Add a message to the conversation history."""
        prefix = "Human: " if is_human else "Assistant: "
        self.conversation_history.append(f"{prefix}{message}")
    
    def get_conversation_history(self) -> List[str]:
        """Get the full conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        return {
            "total_messages": len(self.conversation_history),
            "human_messages": len([msg for msg in self.conversation_history if msg.startswith("Human:")]),
            "assistant_messages": len([msg for msg in self.conversation_history if msg.startswith("Assistant:")]),
            "recent_messages": self.conversation_history[-5:] if self.conversation_history else []
        }
    
    def change_prompt_template(self, prompt_name: str):
        """Change the agent's prompt template and recreate the executor."""
        self.set_prompt_template(prompt_name)
        self._setup_agent_executor()  # Recreate executor with new prompt
    
    def change_model(self, model_name: str):
        """Change the agent's model and recreate the executor."""
        self.set_model(model_name)
        self._setup_agent_executor()  # Recreate executor with new model
    
    def add_tool(self, tool: Union[str, BaseTool]):
        """Add a tool to the agent and recreate the executor."""
        super().add_tool(tool)
        self._setup_agent_executor()  # Recreate executor with new tools
    
    def remove_tool(self, tool_name: str):
        """Remove a tool from the agent and recreate the executor."""
        super().remove_tool(tool_name)
        self._setup_agent_executor()  # Recreate executor with updated tools
    
    def set_max_iterations(self, max_iterations: int):
        """Set the maximum number of iterations and recreate the executor."""
        self.max_iterations = max_iterations
        self._setup_agent_executor()
    
    def set_verbose(self, verbose: bool):
        """Set the verbose mode and recreate the executor."""
        self.verbose = verbose
        self._setup_agent_executor()
    
    def list_available_tools(self) -> List[str]:
        """List all available tool names for this agent."""
        return [tool.name for tool in self.tools]
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get detailed information about available tools."""
        tool_info = {}
        for tool in self.tools:
            # Get args_schema safely
            args_schema = None
            try:
                if hasattr(tool, 'args_schema'):
                    schema = tool.args_schema
                    if hasattr(schema, '__name__'):
                        args_schema = schema.__name__
                    elif isinstance(schema, type):
                        args_schema = schema.__name__
                    else:
                        args_schema = str(type(schema))
            except:
                args_schema = "Unknown"
            
            tool_info[tool.name] = {
                "description": getattr(tool, 'description', 'No description'),
                "type": type(tool).__name__,
                "args_schema": args_schema
            }
        return tool_info
    
    def has_tool(self, tool_name: str) -> bool:
        """Check if the agent has a specific tool."""
        return any(tool.name == tool_name for tool in self.tools)
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the agent."""
        # Get base info but filter out non-serializable objects
        base_info = super().get_agent_info()
        
        # Create a serializable version of the info
        info = {
            "name": self.name,
            "model_name": self.model_name,
            "prompt_name": getattr(self, 'prompt_name', None),
            "max_iterations": self.max_iterations,
            "verbose": self.verbose,
            "has_executor": self.agent_executor is not None,
            "executor_type": "AgentExecutor" if self.agent_executor else "None",
            "conversation_length": len(self.conversation_history),
            "conversation_summary": self.get_conversation_summary(),
            "available_tools": self.list_available_tools(),
            "tool_count": len(self.tools),
            "tool_info": self.get_tool_info()
        }
        
        # Add any serializable base info
        for key, value in base_info.items():
            if key not in info and isinstance(value, (str, int, float, bool, list, dict, type(None))):
                info[key] = value
        
        return info
        
    def __str__(self):
        return json.dumps(self.get_agent_info(), indent=4)
    
    def __repr__(self):
        return self.__str__()
