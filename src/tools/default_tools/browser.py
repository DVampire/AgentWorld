"""Browser tool for interacting with the browser."""

import os
from typing import Type, Dict, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from browser_use import Agent

from src.models import model_manager
from src.utils import assemble_project_path

_BROWSER_TOOL_DESCRIPTION = """Use the browser to interact with the internet to complete the task."""

class BrowserToolArgs(BaseModel):
    task: str = Field(description="The task to complete.")
    base_dir: str = Field(description="The directory to save the files.")

class BrowserTool(BaseTool):
    """A tool for interacting with the browser asynchronously."""
    
    name: str = "browser"
    description: str = _BROWSER_TOOL_DESCRIPTION
    args_schema: Type[BrowserToolArgs] = BrowserToolArgs
    
    model_name: str = Field(
        default="bs-gpt-4.1",
        description="The model to use for the browser."
    )
    
    def __init__(self, model_name: str = "bs-gpt-4.1", **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        
    async def _arun(self, task: str, base_dir: str) -> str:
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
                return f"Error creating browser agent: {str(e)}"

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
                res = f"Task completed but error extracting results: {str(e)}"

            await agent.close()
            return res

        except Exception as e:
            return f"Error in browser tool: {str(e)}"
        
    def _run(self, task: str, base_dir: str) -> str:
        """Interact with the browser synchronously."""
        try:
            return self._arun(task, base_dir)
        except Exception as e:
            return f"Error in browser tool: {str(e)}"
        
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
        }