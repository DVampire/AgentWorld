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

from pydantic import BaseModel, ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.model import model_manager
from src.prompt import prompt_manager
from src.registry import AGENT
from src.session import SessionContext

class EvaluationResult(BaseModel):
    reasoning: str = Field(description="Key reasoning steps and conclusions. Explain what the code did, what was computed, and why the answer is correct (or why the task failed). For multiple-choice tasks, must include explicit analysis of EVERY option (why correct or incorrect) before committing to an answer.")
    answer: str = Field(description="The final answer or result of the execution. Concise and directly usable. Scoped strictly to what the task asks for. For multiple-choice tasks, list all correct options explicitly (e.g. 'A, C' for multi-select; 'B' for single-select).")


_DESCRIPTION = (
    "Coding agent powered by the opencode CLI. Supports Python and R for "
    "data analysis, computation, and scripting tasks. Runs `opencode run \"<task>\"` "
    "inside a session-scoped working directory and returns the full execution output."
)

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
        default=_DESCRIPTION
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
        summary_model_name: Optional[str] = None,
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
        # Model used to summarize the (often very long) opencode output
        self.summary_model_name = summary_model_name

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

            # Evaluate the (often very long) output before returning
            if self.summary_model_name and output:
                eval_result = await self._evaluate(task, output)
                parts = []
                if eval_result.reasoning:
                    parts.append("**Reasoning:**\n" + eval_result.reasoning)
                if eval_result.answer:
                    parts.append("**Answer:** " + eval_result.answer)
                summary = "\n\n".join(parts) if parts else output
            else:
                summary = output

            return AgentResponse(
                success=success,
                message=summary,
                extra=AgentExtra(
                    data={
                        "task": task,
                        "run_dir": run_dir,
                        "returncode": proc.returncode,
                        "raw_output": output,
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

    # ------------------------------------------------------------------
    # Summarize long output
    # ------------------------------------------------------------------

    async def _evaluate(self, task: str, output: str) -> EvaluationResult:
        """Use a lightweight LLM to evaluate the raw opencode output and extract reasoning + answer."""
        logger.info(f"| 📝 Evaluating opencode output ({len(output)} chars) ...")
        try:
            messages = await prompt_manager.get_messages(
                prompt_name="opencode_eval",
                agent_modules={"task": task, "output": output},
            )
            resp = await model_manager(
                model=self.summary_model_name,
                messages=messages,
                response_format=EvaluationResult,
            )
            if resp and resp.extra and hasattr(resp.extra, "parsed_model") and resp.extra.parsed_model:
                result = resp.extra.parsed_model
                logger.info(f"| ✅ Evaluation done.")
                return result
            return EvaluationResult(
                reasoning=resp.message.strip() if resp else "",
                answer="",
            )
        except Exception as exc:
            logger.warning(f"| ⚠️ Evaluation failed, returning raw output: {exc}")
            return EvaluationResult(reasoning="", answer=output[:2000])
