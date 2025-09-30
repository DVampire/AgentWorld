"""Browser tool for interacting with the browser."""

import asyncio
import os
from typing import Type, Dict, Any, Optional
from pydantic import BaseModel, Field
from browser_use import Agent

from src.infrastructures.models import model_manager
from src.utils import assemble_project_path
from src.tools.protocol import tcp
from src.config import config
from src.tools.protocol.tool import BaseTool
from src.tools.protocol.types import ToolResponse
from src.logger import logger

_BROWSER_TOOL_DESCRIPTION = """Use the browser to interact with the internet to complete the task."""

class BrowserToolArgs(BaseModel):
    task: str = Field(description="The task to complete.")
    base_dir: str = Field(description="The directory to save the files.")

@tcp.tool()
class BrowserTool(BaseTool):
    """A tool for interacting with the browser asynchronously."""
    
    name: str = "browser"
    type: str = "Browser"
    description: str = _BROWSER_TOOL_DESCRIPTION
    args_schema: Type[BrowserToolArgs] = BrowserToolArgs
    metadata: Dict[str, Any] = {}
    
    model_name: str = Field(
        default="bs-gpt-4.1",
        description="The model to use for the browser."
    )
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        
        super().__init__(**kwargs)
        
        if model_name is not None:
            self.model_name = model_name
        else:
            if "browser_tool" in config:
                self.model_name = config.browser_tool.get("model_name", "bs-gpt-4.1")
        
        
    async def _arun(self, task: str, base_dir: str) -> ToolResponse:
        agent = None
        try:
            base_dir = assemble_project_path(base_dir)
            os.makedirs(base_dir, exist_ok=True)

            try:
                agent = Agent(
                    task=task,
                    llm=model_manager.get(self.model_name),
                    page_extraction_llm=model_manager.get(self.model_name),
                    file_system_path=base_dir,
                    generate_gif=os.path.join(base_dir, "browser.gif"),
                    save_conversation_path=os.path.join(base_dir, "logs"),
                    max_steps=50,
                    verbose=True,
                )
            except Exception as e:
                return ToolResponse(content=f"Error creating browser agent: {str(e)}")

            history = await agent.run()

            try:
                if hasattr(history, "extracted_content"):
                    contents = history.extracted_content()
                    res = "\n".join(contents) if contents else "No extracted content found"
                elif hasattr(history, "final_result"):
                    res = history.final_result() or "No final result available"
                elif hasattr(history, "history") and history.history:
                    last_step = history.history[-1]
                    res = str(getattr(last_step, "action_results", last_step))
                else:
                    res = "Task completed but no specific results available"
            except Exception as e:
                res = ToolResponse(content=f"Task completed but error extracting results: {str(e)}")

            return res

        except Exception as e:
            return ToolResponse(content=f"Error in browser tool: {str(e)}")
        finally:
            # Ensure proper cleanup
            if agent:
                try:
                    # Close the agent
                    await agent.close()
                except Exception as e:
                    logger.warning(f"Error during browser cleanup: {e}")
        
    def _run(self, task: str, base_dir: str) -> ToolResponse:
        """Execute deep analysis synchronously (fallback)."""
        try:
            # Run async version
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(task, base_dir))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous execution: {e}")
            return ToolResponse(content=f"Error in synchronous execution: {e}")