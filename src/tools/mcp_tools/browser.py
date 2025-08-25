import os
from mcp.types import ToolAnnotations
from mcp.server.fastmcp.server import Context
from pydantic import BaseModel, Field
from browser_use import Agent, ChatOpenAI
from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.tools.mcp_tools.server import mcp_server, MCP_TOOL_ARGS
from src.utils import assemble_project_path

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
        args: BrowserToolInput, The input of the browser tool.

    Returns:
        dict: The result of the task.
    """
    await ctx.info(f"Completing task: {task}")
    
    data_dir = assemble_project_path(data_dir)
    os.makedirs(data_dir, exist_ok=True)    
    
    agent = Agent(
        task=task,
        llm=ChatOpenAI(model="gpt-4.1", 
                       api_key=os.getenv("OPENAI_API_KEY"), 
                       base_url=os.getenv("OPENAI_API_BASE")),
        enable_memory=False,
        page_extraction_llm=ChatOpenAI(model="gpt-4.1", 
                                       api_key=os.getenv("OPENAI_API_KEY"), 
                                       base_url=os.getenv("OPENAI_API_BASE")),
        file_system_path=data_dir
    )
    
    
    history = await agent.run(max_steps=50)
    contents = history.extracted_content()
    res = "\n".join(contents)
    await agent.close()
    
    return res