"""Prompt templates for BrowserUseAgent."""
#这个其实没有在真正执行中被用到
from typing import Any, Dict

from pydantic import ConfigDict, Field

from src.prompt.types import Prompt
from src.registry import PROMPT


AGENT_PROFILE = """
You are a browser execution specialist that completes tasks by controlling a real web browser.
Your objective is to execute the given task faithfully and return practical, verifiable results.
"""

INPUT_CONTRACT = """
<input>
- You will receive a task that may include website goals, expected outputs, and optional file hints.
- Execute the task in browser-use and report what was actually completed.
</input>
"""

EXECUTION_RULES = """
<execution_rules>
- Prioritize correctness over speed.
- If a requested action is blocked, report the blocker explicitly.
- Return concrete findings, not generic statements.
</execution_rules>
"""

OUTPUT_RULES = """
<output_rules>
- Provide concise execution summary.
- Mention critical outcomes and blockers.
- Do not fabricate actions or webpage results.
</output_rules>
"""

SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ input_contract }}
{{ execution_rules }}
{{ output_rules }}
"""

SYSTEM_PROMPT = {
    "name": "browser_use_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for BrowserUseAgent",
    "require_grad": True,
    "template": SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Browser execution agent identity.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_PROFILE,
        },
        "input_contract": {
            "name": "input_contract",
            "type": "system_prompt",
            "description": "Input assumptions and scope.",
            "require_grad": False,
            "template": None,
            "variables": INPUT_CONTRACT,
        },
        "execution_rules": {
            "name": "execution_rules",
            "type": "system_prompt",
            "description": "Execution behavior constraints.",
            "require_grad": True,
            "template": None,
            "variables": EXECUTION_RULES,
        },
        "output_rules": {
            "name": "output_rules",
            "type": "system_prompt",
            "description": "Output expectations.",
            "require_grad": False,
            "template": None,
            "variables": OUTPUT_RULES,
        },
    },
}

AGENT_MESSAGE_PROMPT_TEMPLATE = """
Task:
{{ task }}

{% if files %}
Files:
{{ files }}
{% endif %}

{% if agent_context %}
Agent context:
{{ agent_context }}
{% endif %}
"""

AGENT_MESSAGE_PROMPT = {
    "name": "browser_use_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic task input for BrowserUseAgent",
    "require_grad": False,
    "template": AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "User task for browser execution.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "files": {
            "name": "files",
            "type": "agent_message_prompt",
            "description": "Optional file list for context.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "agent_context": {
            "name": "agent_context",
            "type": "agent_message_prompt",
            "description": "Optional runtime context.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
    },
}


@PROMPT.register_module(force=True)
class BrowserUseSystemPrompt(Prompt):
    """System prompt for BrowserUseAgent."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="browser_use")
    description: str = Field(default="System prompt for BrowserUseAgent")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class BrowserUseAgentMessagePrompt(Prompt):
    """Agent message prompt for BrowserUseAgent."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="browser_use")
    description: str = Field(default="Dynamic task input for BrowserUseAgent")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=AGENT_MESSAGE_PROMPT)

