from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict
from pydantic import Field, ConfigDict


# ===========================================================================
# Query — per-round search query generation
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

QUERY_AGENT_PROFILE = """
You are a helpful assistant that analyzes research tasks and generates optimized search queries.
Your goal is to produce a concise, focused query that maximizes the chance of finding a
complete answer to the research task.
"""

QUERY_RULES = """
<query_rules>
- The search query must be concise and focused (typically 3-8 words).
- Avoid long phrases or complete sentences — keep it short and search-friendly.
- On round 1: focus on the most important keywords and concepts from the task.
- On round > 1: generate a query that finds MISSING information. Use different keywords,
  angles, or related topics compared to previous rounds.
- If an image is provided, analyze it and incorporate relevant visual information.
- Return ONLY the search query — no explanation, no punctuation at the end.
</query_rules>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

QUERY_SYSTEM_PROMPT_TEMPLATE = """
{{ query_agent_profile }}
{{ query_rules }}
"""

QUERY_AGENT_MESSAGE_TEMPLATE = """
Research task: "{{ task }}"

{% if image %}
Image provided: {{ image }}
{% endif %}

Round: {{ round_number }}{% if round_number == "1" %} (initial){% endif %}

Previous search results:
{{ previous_context }}

{% if round_number != "1" %}
Generate a NEW search query that targets information NOT yet found in previous rounds.
{% else %}
Generate an optimized search query for this task.
{% endif %}

Return only the search query, nothing else.
"""

# ---------------------------------------------------------------------------
# Prompt config dicts
# ---------------------------------------------------------------------------

QUERY_SYSTEM_PROMPT = {
    "name": "deep_researcher_query_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for deep researcher query generation",
    "require_grad": True,
    "template": QUERY_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "query_agent_profile": {
            "name": "query_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the query generation component.",
            "require_grad": False,
            "template": None,
            "variables": QUERY_AGENT_PROFILE,
        },
        "query_rules": {
            "name": "query_rules",
            "type": "system_prompt",
            "description": "Rules for generating focused search queries.",
            "require_grad": True,
            "template": None,
            "variables": QUERY_RULES,
        },
    },
}

QUERY_AGENT_MESSAGE_PROMPT = {
    "name": "deep_researcher_query_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Per-round dynamic context for query generation",
    "require_grad": False,
    "template": QUERY_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The research task.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "image": {
            "name": "image",
            "type": "agent_message_prompt",
            "description": "Optional image path.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "round_number": {
            "name": "round_number",
            "type": "agent_message_prompt",
            "description": "Current round number (1-based).",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "previous_context": {
            "name": "previous_context",
            "type": "agent_message_prompt",
            "description": "Plain-text log of previous research rounds.",
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
class DeepResearcherQuerySystemPrompt(Prompt):
    """System prompt for deep researcher query generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_researcher_query")
    description: str = Field(default="System prompt for deep researcher query generation")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=QUERY_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepResearcherQueryAgentMessagePrompt(Prompt):
    """Agent message prompt for deep researcher query generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_researcher_query")
    description: str = Field(default="Per-round dynamic context for query generation")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=QUERY_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Eval — per-round completeness evaluation
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

EVAL_AGENT_PROFILE = """
You are a rigorous research evaluator. Your job is to assess whether a web search summary
provides a complete, sufficient answer to the original research task.
"""

EVAL_RULES = """
<evaluation_rules>
Evaluate the summary against the research task. Consider:
- Does the information directly address the task?
- Is there sufficient detail and depth?
- Are there multiple perspectives or sources mentioned?
- Is the information comprehensive enough to be useful?

Be decisive: mark is_complete=true only when the summary genuinely answers the task.
</evaluation_rules>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT_TEMPLATE = """
{{ eval_agent_profile }}
{{ evaluation_rules }}
"""

EVAL_AGENT_MESSAGE_TEMPLATE = """
Research Task: {{ task }}

Summary from web search:
{{ summary }}

Evaluate whether this summary provides a complete answer to the research task.
"""

# ---------------------------------------------------------------------------
# Prompt config dicts
# ---------------------------------------------------------------------------

EVAL_SYSTEM_PROMPT = {
    "name": "deep_researcher_eval_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for deep researcher completeness evaluation",
    "require_grad": True,
    "template": EVAL_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "eval_agent_profile": {
            "name": "eval_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the evaluation component.",
            "require_grad": False,
            "template": None,
            "variables": EVAL_AGENT_PROFILE,
        },
        "evaluation_rules": {
            "name": "evaluation_rules",
            "type": "system_prompt",
            "description": "Criteria for judging research completeness.",
            "require_grad": True,
            "template": None,
            "variables": EVAL_RULES,
        },
    },
}

EVAL_AGENT_MESSAGE_PROMPT = {
    "name": "deep_researcher_eval_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Per-round dynamic context for completeness evaluation",
    "require_grad": False,
    "template": EVAL_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The research task.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "summary": {
            "name": "summary",
            "type": "agent_message_prompt",
            "description": "Merged search result summary for the current round.",
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
class DeepResearcherEvalSystemPrompt(Prompt):
    """System prompt for deep researcher completeness evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_researcher_eval")
    description: str = Field(default="System prompt for deep researcher completeness evaluation")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=EVAL_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepResearcherEvalAgentMessagePrompt(Prompt):
    """Agent message prompt for deep researcher completeness evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_researcher_eval")
    description: str = Field(default="Per-round context for completeness evaluation")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=EVAL_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Summary — final report summarization
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

SUMMARY_AGENT_PROFILE = """
You are an expert at summarizing research reports. You generate clear, informative summaries
that MUST explicitly state whether answers were found in the first line using
"Answer Found: Yes" or "Answer Found: No".
"""

SUMMARY_RULES = """
<summary_rules>
Generate a summary that:
1. MUST start with a clear statement: "Answer Found: Yes" or "Answer Found: No" (exact format).
2. Provides a concise overview of the key findings.
3. Highlights the most important information discovered.
4. Mentions the number of research rounds conducted.
5. If answer was found: summarize the key answer points.
6. If answer was not found: explain what was discovered and what gaps remain.
</summary_rules>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT_TEMPLATE = """
{{ summary_agent_profile }}
{{ summary_rules }}
"""

SUMMARY_AGENT_MESSAGE_TEMPLATE = """
Research Task: {{ task }}

Research Report:
{{ report_content }}

Generate a comprehensive summary following the rules above.
The first line MUST explicitly state whether the answer was found.
"""

# ---------------------------------------------------------------------------
# Prompt config dicts
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = {
    "name": "deep_researcher_summary_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for deep researcher final summary generation",
    "require_grad": True,
    "template": SUMMARY_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "summary_agent_profile": {
            "name": "summary_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the summary generation component.",
            "require_grad": False,
            "template": None,
            "variables": SUMMARY_AGENT_PROFILE,
        },
        "summary_rules": {
            "name": "summary_rules",
            "type": "system_prompt",
            "description": "Rules for generating the final research summary.",
            "require_grad": True,
            "template": None,
            "variables": SUMMARY_RULES,
        },
    },
}

SUMMARY_AGENT_MESSAGE_PROMPT = {
    "name": "deep_researcher_summary_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for final summary generation",
    "require_grad": False,
    "template": SUMMARY_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The research task.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "report_content": {
            "name": "report_content",
            "type": "agent_message_prompt",
            "description": "Full content of the compiled research report.",
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
class DeepResearcherSummarySystemPrompt(Prompt):
    """System prompt for deep researcher final summary generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_researcher_summary")
    description: str = Field(default="System prompt for deep researcher final summary generation")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=SUMMARY_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepResearcherSummaryAgentMessagePrompt(Prompt):
    """Agent message prompt for deep researcher final summary generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_researcher_summary")
    description: str = Field(default="Dynamic context for final summary generation")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=SUMMARY_AGENT_MESSAGE_PROMPT)
