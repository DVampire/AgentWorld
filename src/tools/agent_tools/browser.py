"""Browser tool for interacting with the browser."""

import asyncio
import os
from typing import Type, Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from browser_use import Agent

from src.infrastructures.models import model_manager
from src.utils import assemble_project_path
from src.tools.protocol import tcp
from src.config import config
from src.tools.protocol.tool import ToolResponse
from src.logger import logger

_BROWSER_TOOL_DESCRIPTION = """Use the browser to interact with the internet to complete the task."""

class BrowserToolArgs(BaseModel):
    task: str = Field(description="The task to complete.")
    base_dir: str = Field(description="The directory to save the files.")

@tcp.tool()
class BrowserTool(BaseTool):
    """A tool for interacting with the browser asynchronously."""
    
    name: str = "browser"
    description: str = _BROWSER_TOOL_DESCRIPTION
    args_schema: Type[BrowserToolArgs] = BrowserToolArgs
    metadata: Dict[str, Any] = {"type": "Browser"}
    
    model_name: str = Field(
        default="bs-gpt-4.1",
        description="The model to use for the browser."
    )
    
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        model_name = model_name or config.browser_tool.get("model_name", "bs-gpt-4.1")
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name
        
    async def _arun(self, task: str, base_dir: str) -> ToolResponse:
        try:
            base_dir = assemble_project_path(base_dir)
            os.makedirs(base_dir, exist_ok=True)

            try:
                agent = Agent(
                    task=task,
                    llm=model_manager.get_model(self.model_name),
                    page_extraction_llm=model_manager.get_model(self.model_name),
                    file_system_path=base_dir,
                    max_steps=20,
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

            await agent.close()
            return res

        except Exception as e:
            return ToolResponse(content=f"Error in browser tool: {str(e)}")
        
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