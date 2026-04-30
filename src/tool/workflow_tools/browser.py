"""Browser tool for interacting with the browser."""

import asyncio
import os
from typing import Optional, Dict, Any
from pydantic import Field, ConfigDict
from browser_use import Agent as BrowserUseRunner
from browser_use import BrowserSession
from browser_use.llm import ChatOpenAI

from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.utils import assemble_project_path, generate_unique_id
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.tool.workflow_tools.reporter import Report
from src.logger import logger
from src.registry import TOOL


_BROWSER_TOOL_DESCRIPTION = """Use the browser to interact with the internet to complete the task.
- If you want to navigate to a search website, bing (https://www.bing.com/) is the best option.
"""

@TOOL.register_module(force=True)
class BrowserTool(Tool):
    """A tool for interacting with the browser asynchronously."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = "browser_tool"
    description: str = _BROWSER_TOOL_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")
    
    model_name: str = Field(
        default="openrouter/gpt-4.1",
        description="The model to use for the browser."
    )
    
    base_dir: str = Field(
        default="workdir/browser",
        description="The base directory to use for the browser."
    )

    browser_start_timeout_sec: int = Field(
        default=120,
        description="Timeout for browser start/launch events in seconds."
    )
    
    def __init__(self, model_name: Optional[str] = None, base_dir: Optional[str] = None, require_grad: bool = False, **kwargs):
        
        super().__init__(require_grad=require_grad, **kwargs)
        
        if model_name is not None:
            self.model_name = model_name
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(self.base_dir)
            
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"| Browser tool base directory: {self.base_dir}")

    async def _run_browser_task(self, task: str, run_id: str, max_steps: int = 50):
        """Run one browser-use task for tool mode."""
        save_dir = os.path.join(self.base_dir, run_id) if self.base_dir else None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        profile_dir = os.path.join(save_dir, "profile") if save_dir else None
        if profile_dir:
            os.makedirs(profile_dir, exist_ok=True)

        gif_path = os.path.join(save_dir, "browser.gif") if save_dir else None
        logs_path = os.path.join(save_dir, "browser.log") if save_dir else None

        runner = None
        try:
            # browser_use defaults these events to 30s, which is often too short on CI/remote machines.
            timeout_str = str(self.browser_start_timeout_sec)
            os.environ.setdefault("TIMEOUT_BrowserStartEvent", timeout_str)
            os.environ.setdefault("TIMEOUT_BrowserLaunchEvent", timeout_str)
            os.environ.setdefault("TIMEOUT_BrowserConnectedEvent", timeout_str)

            browser_session = BrowserSession(
                headless=True,
                chromium_sandbox=False,
                user_data_dir=profile_dir,
                downloads_path=save_dir,
            )

            runner = BrowserUseRunner(
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
                browser_session=browser_session,
                file_system_path=save_dir,
                generate_gif=gif_path,
                save_conversation_path=logs_path,
                max_steps=max_steps,
                verbose=True,
            )
            history = await runner.run()
            try:
                if hasattr(history, "extracted_content"):
                    contents = history.extracted_content()
                    result_message = "\n".join(contents) if contents else "No extracted content found"
                elif hasattr(history, "final_result"):
                    result_message = history.final_result() or "No final result available"
                elif hasattr(history, "history") and history.history:
                    last_step = history.history[-1]
                    result_message = str(getattr(last_step, "action_results", last_step))
                else:
                    result_message = "Task completed but no specific results available"
            except Exception as exc:
                result_message = f"Task completed but failed to extract rich output: {exc}"
            return True, str(result_message), save_dir, gif_path, logs_path
        except Exception as exc:
            logger.error(f"| ❌ browser tool runtime failed: {exc}")
            return False, f"Error in browser runtime: {exc}", save_dir, gif_path, logs_path
        finally:
            if runner is not None:
                try:
                    maybe_result = runner.close()
                    if asyncio.iscoroutine(maybe_result):
                        await maybe_result
                except Exception as exc:
                    logger.warning(f"| ⚠️ browser tool close failed: {exc}")
                    
    async def __call__(self, task: str, **kwargs) -> ToolResponse:
        """Use the browser to interact with the internet to complete the task.

        Args:
            task (str): The task to complete.
        """
        try:
            logger.info(f"| 🌐 Starting browser task: {task}")
            
            # Generate unique id for this browser task
            id = generate_unique_id(prefix="browser")
            
            # Create file path for markdown report
            md_filename = f"{id}.md"
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
            
            # Execute browser runtime with unique id
            success, result_message, save_dir, gif_path, logs_path = await self._run_browser_task(
                task=task,
                run_id=id,
                max_steps=kwargs.get("max_steps", 50),
            )
            
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
                            "result": result_message,
                            "save_dir": save_dir,
                            "gif_path": gif_path,
                            "logs_path": logs_path,
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
        
