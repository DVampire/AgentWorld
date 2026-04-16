from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict
from pydantic import Field, ConfigDict


# ===========================================================================
# Evaluate System Prompt
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

EVALUATE_AGENT_PROFILE = """
You are the Evaluation module of a Planning Agent.
Your ONLY job is to decide whether the original task has been fully and correctly
completed based on the execution history.
"""

EVALUATE_RULES = """
<evaluate_rules>
- Read <task> and <execution_history> carefully.
- The task is done only if a concrete, correct answer or artefact has been produced by a sub-agent.
- If any sub-agent failed, produced an incomplete result, or further steps are required, the task is NOT done.
- Do NOT plan next steps here — only evaluate completion.
</evaluate_rules>
"""

EVALUATE_OUTPUT = """
<output>
You must ALWAYS respond with a valid JSON in this exact format.
DO NOT add any other text like "```json" or "```" or anything else:

{
  "reasoning": "Step-by-step evaluation: what has been done, what is missing, and the final verdict.",
  "is_done": false
}

When the task is complete:
{
  "reasoning": "...",
  "is_done": true
}
</output>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

EVALUATE_SYSTEM_PROMPT_TEMPLATE = """
{{ evaluate_agent_profile }}
{{ evaluate_rules }}
{{ evaluate_output }}
"""

# ---------------------------------------------------------------------------
# Prompt config dict
# ---------------------------------------------------------------------------

EVALUATE_SYSTEM_PROMPT = {
    "name": "planning_agent_evaluate_system_prompt",
    "type": "system_prompt",
    "description": "Evaluation step: decide if the task is fully complete",
    "require_grad": True,
    "template": EVALUATE_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "evaluate_agent_profile": {
            "name": "evaluate_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the evaluation module.",
            "require_grad": False,
            "template": None,
            "variables": EVALUATE_AGENT_PROFILE,
        },
        "evaluate_rules": {
            "name": "evaluate_rules",
            "type": "system_prompt",
            "description": "Rules for evaluating task completion.",
            "require_grad": True,
            "template": None,
            "variables": EVALUATE_RULES,
        },
        "evaluate_output": {
            "name": "evaluate_output",
            "type": "system_prompt",
            "description": "Output format for the evaluation step.",
            "require_grad": False,
            "template": None,
            "variables": EVALUATE_OUTPUT,
        },
    },
}

# ---------------------------------------------------------------------------
# Class definition
# ---------------------------------------------------------------------------

@PROMPT.register_module(force=True)
class PlanningEvaluateSystemPrompt(Prompt):
    """Evaluation step system prompt — decides if the task is complete."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="planning_agent_evaluate")
    description: str = Field(default="Evaluation step: decide if the task is complete")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=EVALUATE_SYSTEM_PROMPT)


# ===========================================================================
# Complete System Prompt
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

COMPLETE_AGENT_PROFILE = """
You are the Completion module of a Planning Agent.
The task has been evaluated as complete. Your ONLY job is to extract and format
the final answer from the execution history.
"""

COMPLETE_RULES = """
<complete_rules>
- Read <task> and <execution_history> to extract the answer produced by sub-agents.
- Put the final answer in the `final_result` field. It must be a number OR as few words as possible OR a comma-separated list of numbers and/or strings.
- ALWAYS adhere to any formatting instructions specified in the original task (e.g., alphabetization, sequencing, units, rounding, decimal places, etc.).
- Pay attention to extracting key stage names, personal names, and location names when the task requires them.
- If asked for a number: express it numerically (digits, not words), no commas, and DO NOT include units such as $ or % unless explicitly required.
- If asked for a string: no articles or abbreviations (e.g. for cities) unless required; no trailing punctuation (., !, ?).
- If asked for a comma-separated list: apply the above rules per element depending on whether each element is a number or string.
- If the answer is a mathematical expression, symbol, or set, output it in LaTeX notation wrapped in `$...$` delimiters (e.g. `$x^{2}+1$`). Do NOT strip the `$` delimiters and do NOT simplify to plain text.
- If the answer cannot be determined: set `final_result` to exactly `Unable to determine`.
</complete_rules>
"""

COMPLETE_EXAMPLES = """
<examples>
<good_examples>
{"reasoning": "The sub-agent identified Paris as the capital.", "final_result": "Paris"}
{"reasoning": "A hexagon has 6 sides.", "final_result": "6"}
{"reasoning": "The answer is option B.", "final_result": "B"}
{"reasoning": "Three countries: Germany, France, Italy.", "final_result": "France, Germany, Italy"}
</good_examples>

<bad_examples>
{"reasoning": "...", "final_result": "Z"} BAD: plain text instead of LaTeX, should be "\\mathbb{Z}"
{"reasoning": "...", "final_result": "1 + 3x + ..."} BAD: missing $...$ delimiters, should be "$1 + 3x + ...$"
</bad_examples>
</examples>
"""

COMPLETE_OUTPUT = """
<output>
You must ALWAYS respond with a valid JSON in this exact format.
DO NOT add any other text like "```json" or "```" or anything else:

{
  "reasoning": "Brief justification for the chosen answer.",
  "final_result": "The concise final answer extracted from the execution history, adhering to <complete_rules>."
}
</output>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

COMPLETE_SYSTEM_PROMPT_TEMPLATE = """
{{ complete_agent_profile }}
{{ complete_rules }}
{{ complete_examples }}
{{ complete_output }}
"""

# ---------------------------------------------------------------------------
# Prompt config dict
# ---------------------------------------------------------------------------

COMPLETE_SYSTEM_PROMPT = {
    "name": "planning_agent_complete_system_prompt",
    "type": "system_prompt",
    "description": "Completion step: extract a concise final answer",
    "require_grad": True,
    "template": COMPLETE_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "complete_agent_profile": {
            "name": "complete_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the completion module.",
            "require_grad": False,
            "template": None,
            "variables": COMPLETE_AGENT_PROFILE,
        },
        "complete_rules": {
            "name": "complete_rules",
            "type": "system_prompt",
            "description": "Rules for formatting the final concise answer.",
            "require_grad": True,
            "template": None,
            "variables": COMPLETE_RULES,
        },
        "complete_examples": {
            "name": "complete_examples",
            "type": "system_prompt",
            "description": "Good and bad examples of final answer formatting.",
            "require_grad": False,
            "template": None,
            "variables": COMPLETE_EXAMPLES,
        },
        "complete_output": {
            "name": "complete_output",
            "type": "system_prompt",
            "description": "Output format for the completion step.",
            "require_grad": False,
            "template": None,
            "variables": COMPLETE_OUTPUT,
        },
    },
}

# ---------------------------------------------------------------------------
# Class definition
# ---------------------------------------------------------------------------

@PROMPT.register_module(force=True)
class PlanningCompleteSystemPrompt(Prompt):
    """Completion step system prompt — extracts a concise final answer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="planning_agent_complete")
    description: str = Field(default="Completion step: extract concise final answer")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=COMPLETE_SYSTEM_PROMPT)


# ===========================================================================
# Continue System Prompt
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

CONTINUE_AGENT_PROFILE = """
You are the Planning module of a Planning Agent — the central orchestrator on the AgentBus.
The task is NOT yet complete. Your ONLY job is to decide which sub-agents to dispatch next.
You do NOT execute tools or call agents yourself. The bus handles all dispatching.
"""

CONTINUE_LANGUAGE_SETTINGS = """
<language_settings>
- Default working language: **English**
- Always respond in the same language as the user request
</language_settings>
"""

CONTINUE_PLANNING_RULES = """
<planning_rules>
**Task Decomposition**
- Decompose remaining work into the smallest meaningful units; each maps to one agent call.

**Concurrency**
- Agents listed in one round's dispatches run concurrently — group independent sub-tasks to maximise parallelism.
- Only serialise (separate rounds) when one sub-task depends on another's output.

**Agent Selection**
- Choose agents by exact name from <available_agents>.
- Write a clear, self-contained task string for each dispatch so the agent has all context it needs.
- Use the right agent for the job:
  - `deep_researcher_light_agent` — web search and online information retrieval tasks.
  - `deep_analyzer_light_agent` — in-depth analysis of images, data, or complex reasoning tasks.
  - `opencode_agent` — computation, coding, scripting, or any task that requires running code.

**File Passing**
- If the original task includes files (images, documents, etc.) listed in <files>, pass the relevant files to the sub-agents that need them via the `files` field of each dispatch.
- When all or most sub-agents need the same files (e.g. an image to analyse), include those files in every relevant dispatch.
</planning_rules>
"""

CONTINUE_OUTPUT = """
<output>
You must ALWAYS respond with a valid JSON in this exact format.
DO NOT add any other text like "```json" or "```" or anything else:

{
  "reasoning": "What has been done, what remains, why these agents and tasks are next.",
  "plan_update": "Updated one-line description of the overall plan.",
  "dispatches": [
    {"agent_name": "exact_agent_name", "task": "Clear sub-task description", "files": ["path/or/url/if/needed"]}
  ]
}

dispatches must be non-empty.
</output>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

CONTINUE_SYSTEM_PROMPT_TEMPLATE = """
{{ continue_agent_profile }}
{{ continue_language_settings }}
{{ continue_planning_rules }}
{{ continue_output }}
"""

# ---------------------------------------------------------------------------
# Prompt config dict
# ---------------------------------------------------------------------------

CONTINUE_SYSTEM_PROMPT = {
    "name": "planning_agent_continue_system_prompt",
    "type": "system_prompt",
    "description": "Continuation step: plan the next round of dispatches",
    "require_grad": True,
    "template": CONTINUE_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "continue_agent_profile": {
            "name": "continue_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the planning/continuation module.",
            "require_grad": False,
            "template": None,
            "variables": CONTINUE_AGENT_PROFILE,
        },
        "continue_language_settings": {
            "name": "continue_language_settings",
            "type": "system_prompt",
            "description": "Language preferences.",
            "require_grad": False,
            "template": None,
            "variables": CONTINUE_LANGUAGE_SETTINGS,
        },
        "continue_planning_rules": {
            "name": "continue_planning_rules",
            "type": "system_prompt",
            "description": "Rules for task decomposition, concurrency, and agent selection.",
            "require_grad": True,
            "template": None,
            "variables": CONTINUE_PLANNING_RULES,
        },
        "continue_output": {
            "name": "continue_output",
            "type": "system_prompt",
            "description": "Output format for the continuation step.",
            "require_grad": False,
            "template": None,
            "variables": CONTINUE_OUTPUT,
        },
    },
}

# ---------------------------------------------------------------------------
# Class definition
# ---------------------------------------------------------------------------

@PROMPT.register_module(force=True)
class PlanningContinueSystemPrompt(Prompt):
    """Continuation step system prompt — plans the next round of dispatches."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="planning_agent_continue")
    description: str = Field(default="Continuation step: plan next dispatches")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=CONTINUE_SYSTEM_PROMPT)


# ===========================================================================
# Agent Message Prompt  (shared context for all three steps)
# ===========================================================================

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

AGENT_MESSAGE_PROMPT_TEMPLATE = """
<task>
{{ task }}
</task>

{% if files %}
<files>
{% for f in files %}
- {{ f }}
{% endfor %}
</files>
{% endif %}

<available_agents>
{{ agent_contract }}
</available_agents>

<round_info>
Round {{ round_number }} of {{ max_rounds }}.
</round_info>

<execution_history>
{{ execution_history }}
</execution_history>
"""

# ---------------------------------------------------------------------------
# Prompt config dict
# ---------------------------------------------------------------------------

AGENT_MESSAGE_PROMPT = {
    "name": "planning_agent_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Shared per-round context (task, agents, history) for all planning steps",
    "require_grad": False,
    "template": AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The original task description.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "files": {
            "name": "files",
            "type": "agent_message_prompt",
            "description": "Optional list of file paths or URLs attached to the task.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "agent_contract": {
            "name": "agent_contract",
            "type": "agent_message_prompt",
            "description": "Available agents and their descriptions.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "round_number": {
            "name": "round_number",
            "type": "agent_message_prompt",
            "description": "Current planning round (1-based).",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "max_rounds": {
            "name": "max_rounds",
            "type": "agent_message_prompt",
            "description": "Maximum allowed rounds.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "execution_history": {
            "name": "execution_history",
            "type": "agent_message_prompt",
            "description": "Plain-text log of all completed rounds.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
    },
}

# ---------------------------------------------------------------------------
# Class definition
# ---------------------------------------------------------------------------

@PROMPT.register_module(force=True)
class PlanningAgentMessagePrompt(Prompt):
    """Shared agent message prompt — per-round context for all planning steps."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="planning_agent")
    description: str = Field(default="Shared per-round context for all planning steps")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=AGENT_MESSAGE_PROMPT)
