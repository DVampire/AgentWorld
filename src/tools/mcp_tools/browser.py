import os
from mcp.types import ToolAnnotations
from mcp.server.fastmcp.server import Context
from pydantic import BaseModel, Field
from browser_use import Agent, ChatOpenAI
from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.tools.mcp_tools.server import mcp_server
from src.utils import assemble_project_path

_BROWSER_TOOL_DESCRIPTION = """
Use the browser to interact with the internet to complete the task.
"""

class BrowserToolInput(BaseModel):
    task: str = Field(description="The task to complete.")
    file_system_path: str = Field(description="The file system path to save the files. ")

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
            "file_system_path": {
                "type": "text",
                "label": "File System Path",
                "placeholder": "Enter a file system path",
            }
        },
    ),  
)
async def browser(args: BrowserToolInput,
                  ctx: Context) -> str:
    """
    Use the browser to interact with the internet to complete the task.

    Args:
        args: BrowserToolInput, The input of the browser tool.

    Returns:
        dict: The result of the task.
    """
    await ctx.info(f"Completing task: {args.task}")
    
    file_system_path = assemble_project_path(args.file_system_path)
    os.makedirs(file_system_path, exist_ok=True)
    
    agent = Agent(
        task=args.task,
        llm=ChatOpenAI(model="gpt-4.1", 
                       api_key=os.getenv("OPENAI_API_KEY"), 
                       base_url=os.getenv("OPENAI_API_BASE")),
        enable_memory=False,
        page_extraction_llm=ChatOpenAI(model="gpt-4.1", 
                                       api_key=os.getenv("OPENAI_API_KEY"), 
                                       base_url=os.getenv("OPENAI_API_BASE")),
        file_system_path=file_system_path
    )
    
    
    history = await agent.run(max_steps=50)
    contents = history.extracted_content()
    res = "\n".join(contents)
    await agent.close()
    
    return res