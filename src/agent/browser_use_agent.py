"""BrowserUseAgent - browser-use execution as a standalone agent."""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from browser_use import Agent as BrowserUseRunner
from browser_use import BrowserSession
from browser_use.llm import ChatOpenAI
from pydantic import ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.registry import AGENT
from src.session import SessionContext
from src.tool.workflow_tools.reporter import Report
from src.utils import assemble_project_path, generate_unique_id


_BROWSER_AGENT_DESCRIPTION = """Use browser-use to operate real webpages and complete browser tasks.
- Best for navigation-heavy web tasks that require iterative page interaction.
- Prefer explicit goals in the task (what page/action/result you want)."""


@AGENT.register_module(force=True)
class BrowserUseAgent(Agent):
    """A standalone agent wrapper for browser-use runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="browser_use_agent")
    description: str = Field(default=_BROWSER_AGENT_DESCRIPTION)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)

    model_name: str = Field(default="openrouter/gpt-4.1")
    base_dir: str = Field(default="workdir/browser")
    max_browser_steps: int = Field(default=50)
    browser_start_timeout_sec: int = Field(default=120)

    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = "browser_use",
        memory_name: Optional[str] = None,
        base_dir: Optional[str] = None,
        max_browser_steps: int = 50,
        browser_start_timeout_sec: int = 120,
        require_grad: bool = False,
        **kwargs,
    ):
        kwargs.setdefault("use_memory", False)
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name or self.model_name,
            prompt_name=prompt_name,
            memory_name=memory_name,
            max_steps=1,
            use_todo=False,
            require_grad=require_grad,
            **kwargs,
        )
        self.max_browser_steps = max_browser_steps
        self.browser_start_timeout_sec = browser_start_timeout_sec
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(os.path.join(workdir, "browser"))
        os.makedirs(self.base_dir, exist_ok=True)

    async def _run_browser_task(self, task: str, run_id: str):
        """Run one browser-use task for agent mode."""
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
            # browser_use defaults BrowserStart/Launch timeout to 30s, which can be too short.
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
                max_steps=self.max_browser_steps,
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
            logger.error(f"| ❌ browser agent runtime failed: {exc}")
            return False, f"Error in browser runtime: {exc}", save_dir, gif_path, logs_path
        finally:
            if runner is not None:
                try:
                    maybe_result = runner.close()
                    if asyncio.iscoroutine(maybe_result):
                        await maybe_result
                except Exception as exc:
                    logger.warning(f"| ⚠️ browser agent close failed: {exc}")

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs,
    ) -> AgentResponse:
        """Execute one browser-use task and return report-oriented agent output."""
        logger.info(f"| 🌐 BrowserUseAgent starting task: {task}")
        ctx = kwargs.get("ctx") or SessionContext()

        task_payload = task
        if files:
            task_payload += "\n\nAttached files:\n" + "\n".join(f"- {f}" for f in files)

        run_id = generate_unique_id(prefix="browser_agent")
        md_filename = f"{run_id}.md"
        file_path = os.path.join(self.base_dir, md_filename)

        report = Report(
            title="Browser Task Report",
            model_name=self.model_name,
            report_file_path=file_path,
        )

        try:
            await report.add_item(f"## Browser Task\n\n{task_payload}\n\n")

            success, result_message, save_dir, gif_path, logs_path = await self._run_browser_task(
                task=task_payload,
                run_id=run_id,
            )

            await report.add_item(f"## Browser Execution Result\n\n{result_message}\n\n")
            await report.complete()

            if success:
                message = (
                    f"Browser task completed successfully!\n\n"
                    f"Task: {task}\n\n"
                    f"Result: {result_message}\n\n"
                    f"Report saved to: {file_path}"
                )
            else:
                message = (
                    f"Browser task failed.\n\n"
                    f"Task: {task}\n\n"
                    f"Result: {result_message}\n\n"
                    f"Report saved to: {file_path}"
                )

            return AgentResponse(
                success=success,
                message=message,
                extra=AgentExtra(
                    file_path=file_path,
                    data={
                        "task": task,
                        "session_id": ctx.id,
                        "result": result_message,
                        "file_path": file_path,
                        "save_dir": save_dir,
                        "gif_path": gif_path,
                        "logs_path": logs_path,
                    },
                ),
            )
        except Exception as exc:
            logger.error(f"| ❌ BrowserUseAgent error: {exc}", exc_info=True)
            return AgentResponse(success=False, message=f"Error in browser agent: {exc}")
