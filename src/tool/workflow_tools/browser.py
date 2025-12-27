"""Browser tool for interacting with the browser."""

import os
import uuid
from typing import Optional, Dict, Any
from pydantic import Field, ConfigDict
from browser_use import Agent
from browser_use.llm import ChatOpenAI

from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.utils import assemble_project_path
from src.utils import file_lock
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.tool.workflow_tools.report import Report
from src.logger import logger
from src.registry import TOOL


_BROWSER_TOOL_DESCRIPTION = """Use the browser to interact with the internet to complete the task.
- If you want to navigate to a search website, bing (https://www.bing.com/) is the best option.
"""

@TOOL.register_module(force=True)
class BrowserTool(Tool):
    """A tool for interacting with the browser asynchronously."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = "browser"
    description: str = _BROWSER_TOOL_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    
    model_name: str = Field(
        default="openrouter/gpt-4.1",
        description="The model to use for the browser."
    )
    
    base_dir: str = Field(
        default="workdir/browser",
        description="The base directory to use for the browser."
    )
    
    def __init__(self, model_name: Optional[str] = None, base_dir: Optional[str] = None, **kwargs):
        
        super().__init__(**kwargs)
        
        if model_name is not None:
            self.model_name = model_name
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(self.base_dir)
            
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| Browser tool base directory: {self.base_dir}")
    
    async def _call_agent(self, task: str, **kwargs) -> ToolResponse:
        """Use the browser to interact with the internet to complete the task.

        Args:
            browser (Browser): The browser to use.
            task (str): The task to complete.
        """
        agent = None
        try:
            try:
                
                agent = Agent(
                    task=task,
                    llm=ChatOpenAI(
                        model=self.model_name.split("/")[-1],
                        base_url=os.getenv("OPENROUTER_API_BASE"),
                        api_key=os.getenv("OPENROUTER_API_KEY"),
                    ),
                    page_extraction_llm=ChatOpenAI(
                        model=self.model_name.split("/")[-1],
                        base_url=os.getenv("OPENROUTER_API_BASE"),
                        api_key=os.getenv("OPENROUTER_API_KEY"),
                    ),
                    file_system_path=self.base_dir if self.base_dir else None,
                    generate_gif=os.path.join(self.base_dir, "browser.gif") if self.base_dir else None,
                    save_conversation_path=os.path.join(self.base_dir, "logs") if self.base_dir else None,
                    max_steps=50,
                    verbose=True,
                )
            except Exception as e:
                return ToolResponse(success=False, message=f"Error creating browser agent: {str(e)}")

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
                res = ToolResponse(success=False, message=f"Task completed but error extracting results: {str(e)}")

            # Close the agent
            agent.close()
            
            return res

        except Exception as e:
            # Close the agent
            agent.close()
            
            return ToolResponse(success=False, message=f"Error in browser tool: {str(e)}")
                    
    
    async def __call__(self, task: str, **kwargs) -> ToolResponse:
        """Use the browser to interact with the internet to complete the task.

        Args:
            task (str): The task to complete.
        """
        try:
            logger.info(f"🌐 Starting browser task: {task}")
            
            # Create file path for markdown report
            md_filename = f"browser_{uuid.uuid4().hex[:8]}.md"
            file_path = os.path.join(self.base_dir, md_filename) if self.base_dir else None
            
            # Initialize Report instance
            report = Report(
                title="Browser Task Report",
                model_name=self.model_name,
                report_file_path=file_path
            )
            
            # Add initial task information
            task_content = f"## Browser Task\n\n{task}\n\n"
            await report.add_item(task_content)
            
            # Execute browser agent
            response = await self._call_agent(task=task, **kwargs)
            
            # Extract result message
            if isinstance(response, ToolResponse):
                result_message = response.message
                success = response.success
            else:
                result_message = str(response)
                success = True
            
            # Add result to report
            result_content = f"## Browser Execution Result\n\n{result_message}\n\n"
            await report.add_item(result_content)
            
            # Generate final report
            if file_path:
                final_report_content = await report.complete()
                logger.info(f"✅ Browser report saved to: {file_path}")
                
                message = f"Browser task completed successfully!\n\nTask: {task}\n\nResult: {result_message}\n\nReport saved to: {file_path}"
                
                return ToolResponse(
                    success=success,
                    message=message,
                    extra=ToolExtra(
                        file_path=file_path,
                        data={
                            "task": task,
                            "file_path": file_path,
                            "result": result_message
                        }
                    )
                )
            else:
                message = f"Browser task completed successfully!\n\nTask: {task}\n\nResult: {result_message}"
                
                return ToolResponse(
                    success=success,
                    message=message
                )
                
        except Exception as e:
            logger.error(f"❌ Error in browser tool: {e}")
            return ToolResponse(success=False, message=f"Error in browser tool: {str(e)}")
        