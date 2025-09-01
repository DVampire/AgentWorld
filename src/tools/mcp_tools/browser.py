import os
from mcp.types import ToolAnnotations
from mcp.server.fastmcp.server import Context
from pydantic import BaseModel, Field
from browser_use import Agent, ChatOpenAI
from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.tools.mcp_tools.server import mcp_server, MCP_TOOL_ARGS
from src.utils import assemble_project_path
from src.models import model_manager

_BROWSER_TOOL_DESCRIPTION = """
Use the browser to interact with the internet to complete the task.
"""

class BrowserToolArgs(BaseModel):
    task: str = Field(description="The task to complete.")
    data_dir: str = Field(description="The directory to save the files. ")

MCP_TOOL_ARGS["browser"] = BrowserToolArgs

@mcp_server.tool(
    name="browser",
    title="Browser Tool",
    description=_BROWSER_TOOL_DESCRIPTION,
    annotations=ToolAnnotations(
        category="browser",
        display_name="Browser Tool",
        input_ui={
            "task": {
                "type": "text",
                "label": "Task",
                "placeholder": "Enter a task",
            },
            "data_dir": {
                "type": "text",
                "label": "Data Directory",
                "placeholder": "Enter a data directory",
            }
        },
    ),  
)
async def browser(task: str,
                  data_dir: str,
                  ctx: Context) -> str:
    """
    Use the browser to interact with the internet to complete the task.

    Args:
        task: The task to complete.
        data_dir: The directory to save the files.

    Returns:
        str: The result of the task.
    """
    try:
        await ctx.info(f"Starting browser task: {task}")
        
        data_dir = assemble_project_path(data_dir)
        os.makedirs(data_dir, exist_ok=True)    
        
        # Create browser-use agent
        agent = Agent(
            task=task,
            llm=model_manager.get_model("bs-gpt-5"),
            page_extraction_llm=model_manager.get_model("bs-gpt-5"),
            file_system_path=data_dir,
            max_steps=20,  # Limit steps to avoid long running tasks
            verbose=True
        )
        
        await ctx.info("Running browser agent...")
        
        # Run the agent
        history = await agent.run()
        
        await ctx.info("Browser agent completed, extracting results...")
        
        # Try to get results from history
        try:
            # Try different methods to extract content
            if hasattr(history, 'extracted_content'):
                contents = history.extracted_content()
                if contents:
                    res = "\n".join(contents)
                else:
                    res = "No extracted content found"
            elif hasattr(history, 'final_result'):
                res = history.final_result() or "No final result available"
            elif hasattr(history, 'history') and history.history:
                # Get the last step result
                last_step = history.history[-1]
                if hasattr(last_step, 'action_results'):
                    res = str(last_step.action_results)
                else:
                    res = str(last_step)
            else:
                res = "Task completed but no specific results available"
                
        except Exception as e:
            await ctx.warning(f"Error extracting content: {e}")
            res = f"Task completed but error extracting results: {str(e)}"
        
        # Close the agent
        await agent.close()
        
        await ctx.info("Browser tool completed successfully")
        
        return res
        
    except Exception as e:
        error_msg = f"Error in browser tool: {str(e)}"
        await ctx.error(error_msg)
        return error_msg