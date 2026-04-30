"""Prompt templates for OpencodeAgent.

One prompt covers the pipeline:

  | Prompt              | 用途                          |
  |---------------------|-------------------------------|
  | opencode_eval   | 评估 opencode 执行输出        |
"""

from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict
from pydantic import Field, ConfigDict


# ===========================================================================
# Evaluate — evaluate opencode execution output
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

EVAL_AGENT_PROFILE = """
You are an expert at evaluating code execution results.
You extract the key reasoning and final answer from verbose CLI output.
"""

EVAL_RULES = """
<eval_rules>
**reasoning field** (required):
- Explain what the code did and the key steps that led to the answer.
- Include any intermediate results, algorithm choices, or error recovery that directly affected the outcome.
- If the execution succeeded: explain briefly *why* the answer is correct (key reasoning step).
- If the execution failed: describe what went wrong and what partial result (if any) was obtained.
- Do NOT reproduce raw logs, debug output, or redundant details — record the insight, not the work.
- **Multiple-choice tasks**: if the task presents discrete options (A/B/C/D or similar), the reasoning MUST include an explicit analysis of EVERY option — stating why each is correct or incorrect — before committing to an answer. Do not stop at the first plausible-looking option.

**answer field** (required):
- State the final result or computed value as concisely as possible.
- Preserve exact mathematical notation, qualifiers, and any symbolic form required by the task.
- **Scope the answer strictly to what the task asks for.** Do not append auxiliary context or intermediate values that were not requested.
- **Multiple-choice tasks**: list all correct options explicitly (e.g. "A, C" for multi-select; "B" for single-select). Do not paraphrase the option text unless the task requires it.
- If the task failed, state what went wrong concisely.
</eval_rules>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT_TEMPLATE = """
{{ eval_agent_profile }}
{{ eval_rules }}
"""

EVAL_AGENT_MESSAGE_TEMPLATE = """
Task: {{ task }}

Execution output:
{{ output }}

Evaluate the above execution output and extract the reasoning and final answer.
"""

# ---------------------------------------------------------------------------
# Prompt config dicts
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = {
    "name": "opencode_eval_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for evaluating opencode execution output",
    "require_grad": True,
    "template": EVAL_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "eval_agent_profile": {
            "name": "eval_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the evaluator.",
            "require_grad": False,
            "template": None,
            "variables": EVAL_AGENT_PROFILE,
        },
        "eval_rules": {
            "name": "eval_rules",
            "type": "system_prompt",
            "description": "Rules for evaluating execution output.",
            "require_grad": True,
            "template": None,
            "variables": EVAL_RULES,
        },
    },
}

EVAL_AGENT_MESSAGE_PROMPT = {
    "name": "opencode_eval_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for opencode output evaluation",
    "require_grad": False,
    "template": EVAL_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The original task.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "output": {
            "name": "output",
            "type": "agent_message_prompt",
            "description": "The raw execution output from opencode.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
    },
}

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------


@PROMPT.register_module(force=True)
class OpencodeEvaluateSystemPrompt(Prompt):
    """System prompt for opencode output evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="opencode_eval")
    description: str = Field(default="System prompt for evaluating opencode execution output")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=EVAL_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class OpencodeEvaluateAgentMessagePrompt(Prompt):
    """Agent message prompt for opencode output evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="opencode_eval")
    description: str = Field(default="Dynamic context for opencode output evaluation")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=EVAL_AGENT_MESSAGE_PROMPT)
