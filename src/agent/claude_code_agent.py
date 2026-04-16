"""Claude Code Agent — wraps Anthropic's claude-agent-sdk (Claude Code CLI).

Invokes the Claude Code CLI via the official Python SDK, following the same
pattern as NanoClaw's agent-runner.  Gives access to the full Claude Code
tool suite: Read, Write, Edit, Bash, Glob, Grep, WebSearch, WebFetch, etc.

Requirements:
    pip install claude-agent-sdk          # Python SDK (already installed)
    npm install -g @anthropic-ai/claude-code  # Claude Code CLI binary
    ANTHROPIC_API_KEY environment variable

The agent calls query() and collects the final ResultMessage, identical to
how NanoClaw's container agent-runner works, just without the IPC loop.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.registry import AGENT

_DEFAULT_ALLOWED_TOOLS = [
    "Bash",
    "Read", "Write", "Edit",
    "Glob", "Grep",
    "WebSearch", "WebFetch",
]


@AGENT.register_module(force=True)
class ClaudeCodeAgent(Agent):
    """Coding agent backed by the Claude Code CLI (claude-agent-sdk).

    Uses Anthropic's official Agent SDK to invoke the full Claude Code tool
    suite locally.  Requires the ``claude`` CLI binary and ANTHROPIC_API_KEY.

    Mirrors NanoClaw's agent-runner approach: call query(), iterate messages,
    return the first ResultMessage as the agent's final answer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="claude_code_agent")
    description: str = Field(
        default=(
            "Coding agent powered by the Claude Code CLI (claude-agent-sdk). "
            "Has access to Read, Write, Edit, Bash, Glob, Grep, WebSearch and "
            "WebFetch tools. Best for multi-file coding tasks, debugging, and "
            "tasks that require navigating a real codebase."
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
        max_iterations: int = 30,
        api_key: Optional[str] = None,
        allowed_tools: Optional[List[str]] = None,
        cli_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name or "claude-opus-4-6",
            prompt_name=prompt_name,
            memory_name=memory_name,
            require_grad=require_grad,
            use_memory=False,
            use_todo=False,
            **kwargs,
        )
        self.max_iterations = max_iterations
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.allowed_tools = allowed_tools or _DEFAULT_ALLOWED_TOOLS
        self.cli_path = cli_path

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs,
    ) -> AgentResponse:
        try:
            from claude_agent_sdk import (
                ClaudeAgentOptions,
                CLINotFoundError,
                ResultMessage,
                query,
            )
        except ImportError:
            return AgentResponse(
                success=False,
                message="claude-agent-sdk is not installed. Run: pip install claude-agent-sdk",
            )

        logger.info(f"| 🤖 ClaudeCodeAgent starting: {task[:120]}")

        os.makedirs(self.workdir, exist_ok=True)

        # Build prompt — append file names if provided
        prompt = task
        if files:
            names = [os.path.basename(f) for f in files if os.path.isfile(f)]
            if names:
                prompt += f"\n\nAvailable files in workspace: {', '.join(names)}"

        # Inject API key into the subprocess environment so the CLI picks it up
        env: Dict[str, str] = {}
        if self.api_key:
            env["ANTHROPIC_API_KEY"] = self.api_key

        options = ClaudeAgentOptions(
            cwd=self.workdir,
            allowed_tools=self.allowed_tools,
            permission_mode="bypassPermissions",
            max_turns=self.max_iterations,
            model=self.model_name,
            env=env,
            cli_path=self.cli_path or None,
        )

        result_text = ""
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result_text = message.result or ""
                    # Do NOT break — early exit causes anyio cancel scope errors.
                    # The generator ends naturally after ResultMessage.

        except CLINotFoundError:
            return AgentResponse(
                success=False,
                message=(
                    "Claude Code CLI not found. Install it with:\n"
                    "  npm install -g @anthropic-ai/claude-code"
                ),
            )
        except Exception as exc:
            logger.error(f"| ❌ ClaudeCodeAgent error: {exc}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"ClaudeCodeAgent failed: {exc}",
            )

        logger.info(
            f"| ✅ ClaudeCodeAgent done. Response length: {len(result_text)} chars"
        )
        return AgentResponse(
            success=True,
            message=result_text,
            extra=AgentExtra(data={"task": task, "model": self.model_name}),
        )
