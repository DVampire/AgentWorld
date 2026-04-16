"""OpenCode Agent — wraps the `opencode` CLI tool.

Runs `opencode run "<task>"` inside a session-scoped working directory
(``<workdir>/<ctx.id>``), captures all output, and returns it as the
agent response.

Requirements:
    opencode CLI binary must be installed and available on PATH.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional  # noqa: F401

from pydantic import ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.registry import AGENT
from src.session import SessionContext


@AGENT.register_module(force=True)
class OpencodeAgent(Agent):
    """Coding agent backed by the `opencode` CLI.

    Changes into ``<workdir>/<ctx.id>`` and executes::

        opencode run "<task>"

    The full stdout/stderr output of the process is returned as the
    agent's response message.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="opencode_agent")
    description: str = Field(
        default=(
            "Coding agent powered by the opencode CLI. "
            "Runs `opencode run \"<task>\"` inside a session-scoped working "
            "directory and returns the full execution output."
        )
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)

    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        require_grad: bool = False,
        timeout: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name,
            prompt_name=prompt_name,
            memory_name=memory_name,
            require_grad=require_grad,
            use_memory=False,
            use_todo=False,
            **kwargs,
        )
        # Optional timeout in seconds for the subprocess (None = no timeout)
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        ctx: Optional[SessionContext] = None,
        **kwargs,
    ) -> AgentResponse:
        # Determine the session-scoped working directory
        if ctx is not None and ctx.id:
            run_dir = os.path.join(self.workdir, ctx.id)
        else:
            run_dir = self.workdir

        os.makedirs(run_dir, exist_ok=True)

        logger.info(f"| 🚀 OpenCodeAgent starting in {run_dir}: {task}")

        prompt = "Use the python code to solve the task. \n\nTask:\n" + task

        try:
            cmd = ["opencode", "run"]
            for f in (files or []):
                cmd += ["-f", f]
            cmd.append(prompt)

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=run_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            try:
                output_bytes, _ = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return AgentResponse(
                    success=False,
                    message=f"OpenCodeAgent timed out after {self.timeout}s",
                )

            output = output_bytes.decode("utf-8", errors="replace")
            success = proc.returncode == 0

            if success:
                logger.info(f"| ✅ OpenCodeAgent done. Output length: {len(output)} chars")
            else:
                logger.warning(f"| ⚠️  OpenCodeAgent exited with code {proc.returncode}")

            return AgentResponse(
                success=success,
                message=output,
                extra=AgentExtra(
                    data={
                        "task": task,
                        "run_dir": run_dir,
                        "returncode": proc.returncode,
                    }
                ),
            )

        except FileNotFoundError:
            return AgentResponse(
                success=False,
                message=(
                    "There were issues running OpenCodeAgent: the `opencode` CLI tool was not found. "
                    "Please ensure it is installed and available on your system PATH."
                ),
            )
        except Exception as exc:
            logger.error(f"| ❌ OpenCodeAgent error: {exc}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"OpenCodeAgent failed: {exc}",
            )
