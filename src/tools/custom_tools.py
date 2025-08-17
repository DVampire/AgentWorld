"""Custom tools for agents."""

from typing import Dict, List, Any
from langchain.tools import BaseTool, tool
import requests
import json
import datetime
import asyncio
import aiohttp


@tool
async def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
async def search_web(query: str) -> str:
    """Search the web for information."""
    # This is a placeholder - you would integrate with a real search API
    # Simulate async operation
    await asyncio.sleep(0.1)
    return f"Search results for: {query} (placeholder implementation)"


@tool
async def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        # Safe evaluation of mathematical expressions
        allowed_names = {
            k: v for k, v in __builtins__.items() 
            if k in ['abs', 'round', 'min', 'max', 'sum']
        }
        allowed_names.update({
            'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum
        })
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
async def weather_lookup(city: str) -> str:
    """Get weather information for a city."""
    # This is a placeholder - you would integrate with a weather API
    # Simulate async API call
    await asyncio.sleep(0.2)
    return f"Weather information for {city} (placeholder implementation)"


@tool
async def file_operations(operation: str, file_path: str, content: str = "") -> str:
    """Perform file operations (read, write, append)."""
    try:
        if operation == "read":
            # Use asyncio to read file
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, lambda: open(file_path, 'r').read())
            return content
        elif operation == "write":
            # Use asyncio to write file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: open(file_path, 'w').write(content))
            return f"Successfully wrote to {file_path}"
        elif operation == "append":
            # Use asyncio to append to file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: open(file_path, 'a').write(content))
            return f"Successfully appended to {file_path}"
        else:
            return f"Unknown operation: {operation}"
    except Exception as e:
        return f"Error performing {operation} on {file_path}: {str(e)}"


@tool
async def async_web_request(url: str) -> str:
    """Make an asynchronous web request."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return f"Successfully fetched {url}. Content length: {len(content)} characters"
                else:
                    return f"Failed to fetch {url}. Status: {response.status}"
    except Exception as e:
        return f"Error fetching {url}: {str(e)}"


@tool
async def batch_calculation(expressions: List[str]) -> str:
    """Calculate multiple mathematical expressions concurrently."""
    try:
        # Process calculations concurrently
        tasks = [calculate(expr) for expr in expressions]
        results = await asyncio.gather(*tasks)
        
        formatted_results = []
        for i, (expr, result) in enumerate(zip(expressions, results)):
            formatted_results.append(f"{i+1}. {expr} = {result}")
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error in batch calculation: {str(e)}"


class CustomToolSet:
    """A collection of custom tools."""
    
    def __init__(self):
        self.tools = [
            get_current_time,
            search_web,
            calculate,
            weather_lookup,
            file_operations,
            async_web_request,
            batch_calculation
        ]
    
    def get_tools(self) -> List[BaseTool]:
        """Get all custom tools."""
        return self.tools
    
    def add_tool(self, tool: BaseTool):
        """Add a custom tool."""
        self.tools.append(tool)
    
    def remove_tool(self, tool_name: str):
        """Remove a tool by name."""
        self.tools = [tool for tool in self.tools if tool.name != tool_name]
    
    async def execute_tool_concurrently(self, tool_name: str, *args, **kwargs) -> str:
        """Execute a tool concurrently."""
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if tool:
            return await tool.ainvoke(*args, **kwargs)
        else:
            return f"Tool {tool_name} not found"
    
    async def execute_multiple_tools(self, tool_calls: List[Dict]) -> List[str]:
        """Execute multiple tools concurrently."""
        tasks = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            args = tool_call.get("args", [])
            kwargs = tool_call.get("kwargs", {})
            
            task = self.execute_tool_concurrently(tool_name, *args, **kwargs)
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
