"""Prompt templates for DeepResearcherV3Agent.

Six prompts cover the ThinkOutput-driven research pipeline:

  | Prompt                              | 用途                                                                |
  |-------------------------------------|---------------------------------------------------------------------|
  | deep_researcher_v3                  | 主循环：tool_calling 风格，LLM 通过 ThinkOutput 选工具              |
  | deep_researcher_v3_query            | QueryTool 的 system + agent message prompt（每轮生成搜索 query）    |
  | deep_researcher_v3_llm_search       | LLM 联网搜索（每个模型一份报告）                                    |
  | deep_researcher_v3_page_summary     | 单页内容摘要（Jina 抓取后）                                         |
  | deep_researcher_v3_synthesis        | 合并页面摘要为一份 API 搜索报告（带引用）                           |
  | deep_researcher_v3_eval             | 综合所有报告、检测冲突、评估完整性 → ResearchSummary               |

deep_researcher_v3 的 system prompt 结构与 tool_calling 完全对齐：
  - agent_profile / agent_introduction / language_settings / input
  - agent_context_rules / tool_context_rules / reasoning_rules / output
agent message prompt 结构与 tool_calling 完全对齐：
  - agent_context / tool_context
"""

from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict
from pydantic import Field, ConfigDict


# ===========================================================================
# Main — system prompt (tool_calling style, 5 research tools)
# ===========================================================================

AGENT_PROFILE = """
You are a rigorous multi-round web research agent. You work in iterative steps and use
five internal tools to plan a research strategy, generate targeted search queries,
execute searches, evaluate results, and finish when the answer is confirmed.
"""

AGENT_INTRODUCTION = """
<intro>
You excel at:
- Breaking complex research tasks into focused search rounds
- Generating optimized, targeted search queries for each round
- Running concurrent API search and LLM web searches, then synthesizing results
- Detecting conflicts between sources and resolving them with targeted follow-up queries
- Completing research accurately and efficiently within the step budget
</intro>
"""

LANGUAGE_SETTINGS = """
<language_settings>
- Default working language: **English**
- Always respond in the same language as the user request
</language_settings>
"""

INPUT = """
<input>
- <agent_context>: Your current internal state — task, step info, full agent history, and memory.
- <tool_context>: The five research tools available to you, with their arguments and usage rules.
</input>
"""

AGENT_CONTEXT_RULES = """
<agent_context_rules>
<task_rules>
TASK: This is your ultimate research objective and always remains visible.
- Analyze the task thoroughly before committing to a course of action.
- You must call the `finish` tool in one of three cases:
  - When eval confirms found_answer=true.
  - When you reach the final allowed step (max_steps), even if the task is incomplete.
  - If it is ABSOLUTELY IMPOSSIBLE to continue (e.g. all tools consistently fail).
</task_rules>

<agent_history_rules>
Your full execution history is provided in <agent_history> as a markdown research log.
Each step is recorded as:

## Step N
**Evaluation:** Assessment of last tool call result
**Memory:** What you retained from this step
**Next Goal:** The goal you set for this step
**Thinking:** Your reasoning
**Action:** The tool you called
**Result:** The tool output

Read the history top-to-bottom before deciding the next action.
</agent_history_rules>
</agent_context_rules>
"""

TOOL_CONTEXT_RULES = """
<tool_context_rules>
<tool_use_rules>
You have exactly five tools. You must call exactly ONE tool per step.

**Decision rules — read the research.md history, then pick:**

- No history (very first step) → call `plan` to initialize the todo list and flowchart with the full research strategy.
- After `plan` (no query yet) → call `query` to generate an optimized search query for the first round.
- After `query` (no search yet) → call `search` to execute concurrent API + LLM searches.
- After `search` → call `eval` immediately to judge the search results.
- After `eval`:
  - found_answer=true → call `finish`.
  - found_answer=false, has_conflict=true → call `query` targeting the specific conflict (new angle).
  - found_answer=false, has_conflict=false → call `query` to explore a different angle or fill the gap.
- Loop guard: same conflict across 3+ consecutive query→search→eval cycles → call `finish` with best available answer.
</tool_use_rules>

<available_tools>
You will be provided with the available tools in <tool_context>.
</available_tools>
</tool_context_rules>
"""

REASONING_RULES = """
<reasoning_rules>
You must reason explicitly at every step in your `thinking` block:

- Read the full agent history to understand what has been done and what is still open.
- Evaluate the last action: did it succeed? what did it reveal?
- Apply the decision rules above to identify exactly which tool to call next and why.
- For `query`, write concrete guidance on the search angle and why it's targeted.
- Before calling `finish`, confirm: eval has returned found_answer=true.
</reasoning_rules>
"""

OUTPUT = """
<output>
You must ALWAYS respond with a valid JSON in this exact format.
DO NOT add any other text like "```json" or "```" or anything else:

{
    "thinking": "Structured reasoning applying the decision rules above.",
    "evaluation_previous_goal": "One-sentence assessment of the last step result.",
    "memory": "1-3 sentences capturing what is known and what remains open.",
    "next_goal": "One clear sentence: which tool, and why.",
    "actions": [
        {
            "type": "tool",
            "name": "<plan|query|search|eval|finish>",
            "args": "{}"
        }
    ]
}

- actions must contain exactly ONE action.
- For `search`, `eval`: args must be "{}".
- For `plan`: args must be "{\"steps\": [\"...\", \"...\"], \"reasoning\": \"...\"}".
- For `query`: args must be "{\"guidance\": \"...\"}".
- For `finish`: args must be "{\"reasoning\": \"...\", \"answer\": \"...\"}".
</output>
"""

SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ agent_introduction }}
{{ language_settings }}
{{ input }}
{{ agent_context_rules }}
{{ tool_context_rules }}
{{ reasoning_rules }}
{{ output }}
"""

SYSTEM_PROMPT = {
    "name": "deep_researcher_v3_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for DeepResearcherV3 ThinkOutput loop",
    "require_grad": True,
    "template": SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Agent identity.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_PROFILE,
        },
        "agent_introduction": {
            "name": "agent_introduction",
            "type": "system_prompt",
            "description": "Agent capabilities.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_INTRODUCTION,
        },
        "language_settings": {
            "name": "language_settings",
            "type": "system_prompt",
            "description": "Language settings.",
            "require_grad": False,
            "template": None,
            "variables": LANGUAGE_SETTINGS,
        },
        "input": {
            "name": "input",
            "type": "system_prompt",
            "description": "Input structure description.",
            "require_grad": False,
            "template": None,
            "variables": INPUT,
        },
        "agent_context_rules": {
            "name": "agent_context_rules",
            "type": "system_prompt",
            "description": "Rules for task, history, and memory.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_CONTEXT_RULES,
        },
        "tool_context_rules": {
            "name": "tool_context_rules",
            "type": "system_prompt",
            "description": "Tool selection decision rules.",
            "require_grad": True,
            "template": None,
            "variables": TOOL_CONTEXT_RULES,
        },
        "reasoning_rules": {
            "name": "reasoning_rules",
            "type": "system_prompt",
            "description": "Step-by-step reasoning guidance.",
            "require_grad": True,
            "template": None,
            "variables": REASONING_RULES,
        },
        "output": {
            "name": "output",
            "type": "system_prompt",
            "description": "Output format (ThinkOutput JSON).",
            "require_grad": False,
            "template": None,
            "variables": OUTPUT,
        },
    },
}


# ===========================================================================
# Main — agent message prompt (tool_calling style)
#
# Variables injected by _get_messages / _get_agent_context / _get_tool_context:
#   agent_context — task + step info + agent history + memory (from base Agent)
#   tool_context  — available_tools block (overridden in DeepResearcherV3Agent)
# ===========================================================================

AGENT_MESSAGE_PROMPT_TEMPLATE = """
{{ agent_context }}
{{ tool_context }}
"""

AGENT_MESSAGE_PROMPT = {
    "name": "deep_researcher_v3_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for DeepResearcherV3 ThinkOutput loop",
    "require_grad": False,
    "template": AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": {
        "agent_context": {
            "name": "agent_context",
            "type": "agent_message_prompt",
            "description": "Task, step info, agent history, and memory from base Agent.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "tool_context": {
            "name": "tool_context",
            "type": "agent_message_prompt",
            "description": "The 5 research tools injected by _get_tool_context.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
    },
}


@PROMPT.register_module(force=True)
class DeepResearcherV3SystemPrompt(Prompt):
    """System prompt for DeepResearcherV3 ThinkOutput loop."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_researcher_v3")
    description: str = Field(default="System prompt for DeepResearcherV3 ThinkOutput loop")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepResearcherV3AgentMessagePrompt(Prompt):
    """Agent message prompt for DeepResearcherV3 ThinkOutput loop."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_researcher_v3")
    description: str = Field(default="Dynamic context for DeepResearcherV3 ThinkOutput loop")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Query — per-round search query generation (QueryTool)
#
# Variables:
#   task             — the research task
#   image            — optional image path/URL; empty string if none
#   filter_year      — optional year to bias results; empty string if none
#   previous_context — execution log of prior rounds; empty if round 1
#   guidance         — angle/focus guidance from ThinkOutput
#
# Round 1 (previous_context empty): generate a broad initial query.
# Round N (previous_context has conflict/gap): generate a targeted query
#   to resolve the specific conflict or fill the gap described.
# ===========================================================================

QUERY_AGENT_PROFILE = """
You are a search strategist for a multi-round research system.
Before each search round, you generate an optimized query that maximizes the
chance of finding the information needed to answer the research task.
"""

QUERY_RULES = """
<query_rules>
**Query format**
- Write a short, direct question or statement — one sentence, as concise as possible.
- Strip all filler words and background context. Include only what is essential for finding the answer.
- Examples: "GELU paper original authors" is too short; "Who are the authors of the GELU paper?" is good; "I need to find out who originally authored the GELU activation function paper" is too long.
- Do NOT use keyword fragments. A grammatical sentence gives better semantic search results.

**First round (no prior results)**
- Generate a focused question or statement that captures the core intent of the task.

**Subsequent rounds (prior results exist)**
- You now know what was found and what remains unresolved — be more targeted.
- Follow the guidance provided — it specifies the angle or focus for this round.
- If prior evaluation describes a conflict: generate a query that targets evidence
  to resolve the specific disagreement.
- If prior evaluation describes a gap or missing information: generate a query that
  directly targets that missing piece.
- If prior rounds were simply incomplete: explore a different angle or related topic.
- Use different phrasing compared to previous rounds. Do not repeat a query already tried.

**Academic and technical topics — prioritise specialised sources**
For domain-specific queries, prefer authoritative sources over generic web search:
math/logic - `arxiv.org`, `mathoverflow.net`, `ncatlab.org`;
philosophy - `plato.stanford.edu`, `philpapers.org`;
sciences - `arxiv.org`, `pubmed.ncbi.nlm.nih.gov`, `semanticscholar.org`;
CS/AI/ML - `arxiv.org`, `semanticscholar.org`, `dl.acm.org`;
linguistics - `glottolog.org`, `wals.info`.

If filter_year is provided, bias the query toward recent results from that year.
If an image is provided, incorporate relevant visual information into the query.
Return ONLY the search query — no explanation, no punctuation at the end.
</query_rules>
"""

QUERY_SYSTEM_PROMPT_TEMPLATE = """
{{ query_agent_profile }}
{{ query_rules }}
"""

QUERY_AGENT_MESSAGE_TEMPLATE = """Research task: "{{ task }}"
{% if image %}

Image provided: {{ image }}
{% endif %}
{% if filter_year %}

Prefer results from year: {{ filter_year }}
{% endif %}
{% if guidance %}

Guidance for this round: {{ guidance }}
{% endif %}
{% if previous_context %}

Prior research rounds (conflicts / gaps to resolve):
{{ previous_context }}

Generate a NEW search query that targets information NOT yet found or resolves the above conflict/gap.
{% else %}
Generate an optimized search query for this task.
{% endif %}

Return only the search query, nothing else."""

QUERY_SYSTEM_PROMPT = {
    "name": "deep_researcher_v3_query_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for per-round search query generation",
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
    "name": "deep_researcher_v3_query_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Per-round dynamic context for search query generation",
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
            "description": "Optional image path or URL; empty string if none.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "filter_year": {
            "name": "filter_year",
            "type": "agent_message_prompt",
            "description": "Optional year to bias search results toward; empty string if none.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "previous_context": {
            "name": "previous_context",
            "type": "agent_message_prompt",
            "description": "Execution log of prior research rounds; empty string on round 1.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "guidance": {
            "name": "guidance",
            "type": "agent_message_prompt",
            "description": "Angle/focus guidance from ThinkOutput for this round.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
    },
}


@PROMPT.register_module(force=True)
class DeepResearcherV3QuerySystemPrompt(Prompt):
    """System prompt for per-round search query generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_researcher_v3_query")
    description: str = Field(default="System prompt for per-round search query generation")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=QUERY_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepResearcherV3QueryAgentMessagePrompt(Prompt):
    """Agent message prompt for per-round search query generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_researcher_v3_query")
    description: str = Field(default="Per-round dynamic context for search query generation")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=QUERY_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# LLM Search — prompt for web-search-capable LLMs (one report per model)
#
# Variables:
#   task  — the original research task (context)
#   query — the optimized search query for this round
# ===========================================================================

LLM_SEARCH_AGENT_PROFILE = """
You are a research assistant with web search capability. Your job is to research a query
and return a comprehensive, well-cited summary of the findings.
"""

LLM_SEARCH_RULES = """
<llm_search_rules>
- Research the query thoroughly and provide a comprehensive summary with citations where possible.
- Include the most relevant and up-to-date information.
- Cite your sources inline where possible (e.g., [Source Name](URL)).
- Return your findings as a comprehensive, well-structured summary.
</llm_search_rules>
"""

LLM_SEARCH_SYSTEM_PROMPT_TEMPLATE = """
{{ llm_search_agent_profile }}
{{ llm_search_rules }}
"""

LLM_SEARCH_AGENT_MESSAGE_TEMPLATE = """Original task context: {{ task }}

Search query: {{ query }}

Research the query and return your findings as a comprehensive summary."""

LLM_SEARCH_SYSTEM_PROMPT = {
    "name": "deep_researcher_v3_llm_search_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for LLM-based web search (one report per model)",
    "require_grad": True,
    "template": LLM_SEARCH_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "llm_search_agent_profile": {
            "name": "llm_search_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the LLM search component.",
            "require_grad": False,
            "template": None,
            "variables": LLM_SEARCH_AGENT_PROFILE,
        },
        "llm_search_rules": {
            "name": "llm_search_rules",
            "type": "system_prompt",
            "description": "Rules for conducting and reporting LLM web search.",
            "require_grad": True,
            "template": None,
            "variables": LLM_SEARCH_RULES,
        },
    },
}

LLM_SEARCH_AGENT_MESSAGE_PROMPT = {
    "name": "deep_researcher_v3_llm_search_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Per-round dynamic context for LLM web search",
    "require_grad": False,
    "template": LLM_SEARCH_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The original research task for context.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "query": {
            "name": "query",
            "type": "agent_message_prompt",
            "description": "The optimized search query for this round.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
    },
}


@PROMPT.register_module(force=True)
class DeepResearcherV3LLMSearchSystemPrompt(Prompt):
    """System prompt for LLM-based web search."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_researcher_v3_llm_search")
    description: str = Field(default="System prompt for LLM-based web search (one report per model)")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=LLM_SEARCH_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepResearcherV3LLMSearchAgentMessagePrompt(Prompt):
    """Agent message prompt for LLM-based web search."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_researcher_v3_llm_search")
    description: str = Field(default="Per-round dynamic context for LLM web search")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=LLM_SEARCH_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Page Summary (internal) — per-page content summarization (Jina fetch → LLM)
# ===========================================================================

PAGE_SUMMARY_AGENT_PROFILE = """
You are a precise information extractor. Given a web page and a search query,
your job is to extract only the information relevant to the query.
"""

PAGE_SUMMARY_RULES = """
<page_summary_rules>
- Extract key information relevant to the query in 2-4 concise sentences.
- Prioritize facts, figures, and direct answers over background context.
- Do not include information unrelated to the query.
- If the page contains no relevant information, respond with a single sentence saying so.
</page_summary_rules>
"""

PAGE_SUMMARY_SYSTEM_PROMPT_TEMPLATE = """
{{ page_summary_agent_profile }}
{{ page_summary_rules }}
"""

PAGE_SUMMARY_AGENT_MESSAGE_TEMPLATE = """
Given search query: "{{ query }}"

Page title: {{ title }}
Page content:
{{ content }}

Extract the key information relevant to the query in 2-4 concise sentences.
"""

PAGE_SUMMARY_SYSTEM_PROMPT = {
    "name": "deep_researcher_v3_page_summary_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for per-page content summarization",
    "require_grad": True,
    "template": PAGE_SUMMARY_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "page_summary_agent_profile": {
            "name": "page_summary_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the page summarization component.",
            "require_grad": False,
            "template": None,
            "variables": PAGE_SUMMARY_AGENT_PROFILE,
        },
        "page_summary_rules": {
            "name": "page_summary_rules",
            "type": "system_prompt",
            "description": "Rules for extracting relevant content from a web page.",
            "require_grad": True,
            "template": None,
            "variables": PAGE_SUMMARY_RULES,
        },
    },
}

PAGE_SUMMARY_AGENT_MESSAGE_PROMPT = {
    "name": "deep_researcher_v3_page_summary_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Per-page dynamic context for content summarization",
    "require_grad": False,
    "template": PAGE_SUMMARY_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "query": {
            "name": "query",
            "type": "agent_message_prompt",
            "description": "The search query used to retrieve this page.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "title": {
            "name": "title",
            "type": "agent_message_prompt",
            "description": "The page title.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "content": {
            "name": "content",
            "type": "agent_message_prompt",
            "description": "The fetched page content (markdown).",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
    },
}


@PROMPT.register_module(force=True)
class DeepResearcherV3PageSummarySystemPrompt(Prompt):
    """System prompt for per-page content summarization."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_researcher_v3_page_summary")
    description: str = Field(default="System prompt for per-page content summarization")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=PAGE_SUMMARY_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepResearcherV3PageSummaryAgentMessagePrompt(Prompt):
    """Agent message prompt for per-page content summarization."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_researcher_v3_page_summary")
    description: str = Field(default="Per-page dynamic context for content summarization")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=PAGE_SUMMARY_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Synthesis (internal) — combine page summaries into a coherent report
# ===========================================================================

SYNTHESIS_AGENT_PROFILE = """
You are an expert research synthesizer. Given multiple web page summaries retrieved
for a search query, your job is to integrate them into a single, coherent research summary.
"""

SYNTHESIS_RULES = """
<synthesis_rules>
- Integrate the key findings from all sources into a coherent, well-structured summary.
- Prioritize information that directly answers the search query.
- Resolve contradictions between sources by noting the disagreement.
- Do not list sources inline — they will be appended as a reference list separately.
- Do not introduce information not present in the provided summaries.
</synthesis_rules>
"""

SYNTHESIS_SYSTEM_PROMPT_TEMPLATE = """
{{ synthesis_agent_profile }}
{{ synthesis_rules }}
"""

SYNTHESIS_AGENT_MESSAGE_TEMPLATE = """
Search query: "{{ query }}"

The following are summaries from {{ num_sources }} web pages retrieved for this query:

{{ sources }}

Synthesize these into a single, comprehensive, well-structured research summary.
Integrate the key findings coherently. Do not list sources inline —
they will be appended as a reference list separately.
"""

SYNTHESIS_SYSTEM_PROMPT = {
    "name": "deep_researcher_v3_synthesis_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for synthesizing page summaries into a research report",
    "require_grad": True,
    "template": SYNTHESIS_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "synthesis_agent_profile": {
            "name": "synthesis_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the synthesis component.",
            "require_grad": False,
            "template": None,
            "variables": SYNTHESIS_AGENT_PROFILE,
        },
        "synthesis_rules": {
            "name": "synthesis_rules",
            "type": "system_prompt",
            "description": "Rules for synthesizing multiple page summaries.",
            "require_grad": True,
            "template": None,
            "variables": SYNTHESIS_RULES,
        },
    },
}

SYNTHESIS_AGENT_MESSAGE_PROMPT = {
    "name": "deep_researcher_v3_synthesis_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for page summary synthesis",
    "require_grad": False,
    "template": SYNTHESIS_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "query": {
            "name": "query",
            "type": "agent_message_prompt",
            "description": "The search query whose results are being synthesized.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "num_sources": {
            "name": "num_sources",
            "type": "agent_message_prompt",
            "description": "Number of source pages being synthesized.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "sources": {
            "name": "sources",
            "type": "agent_message_prompt",
            "description": "Pre-formatted list of [i] title (url)\\nsummary entries.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
    },
}


@PROMPT.register_module(force=True)
class DeepResearcherV3SynthesisSystemPrompt(Prompt):
    """System prompt for page summary synthesis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_researcher_v3_synthesis")
    description: str = Field(default="System prompt for synthesizing page summaries into a research report")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=SYNTHESIS_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepResearcherV3SynthesisAgentMessagePrompt(Prompt):
    """Agent message prompt for page summary synthesis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_researcher_v3_synthesis")
    description: str = Field(default="Dynamic context for page summary synthesis")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=SYNTHESIS_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Eval — synthesize all search reports, detect conflicts, assess completeness
#
# Variables:
#   task     — the research task
#   previous — execution log of prior rounds; empty string on round 1
#   content  — all search reports from this round, each prefixed with
#              **[api_search]** or **[model-name]**
#
# Input format:
#   **[firecrawl_search]**
#   <synthesized report with citations>
#
#   **[model-name]**
#   <llm search report>
#   ...
# ===========================================================================

EVAL_AGENT_PROFILE = """
You are a rigorous research evaluator. You receive reports from multiple search sources
(API search and one or more LLM search models), synthesize their findings, detect any
contradictions between them, and determine whether the research task has been fully answered.
"""

EVAL_RULES = """
<eval_rules>
- Each source's report is prefixed with **[source-name]** (e.g., **[firecrawl_search]**, **[model-name]**).
- Check whether the sources agree or contradict each other on key points relevant to the task.

**reasoning field** (required):
- Capture the key conclusions and critical logic from the search results.
- Always name the source(s) that provided each key finding, using the label in the report
  header (e.g. "**[google_lens_search]** identified X", "**[firecrawl_search]** found Y").
- If sources agree: state the shared finding and strongest supporting evidence.
- If sources conflict: describe the exact disagreement and what evidence would resolve it.
- If found_answer is true: explain briefly *why* the answer is correct (key evidence/logic).
- Do NOT reproduce all the search content — record the insight, not the report.
- **Multiple-choice tasks**: if the task presents discrete options (A/B/C/D or similar), the reasoning MUST include an explicit analysis of EVERY option — stating why each is correct or incorrect — before committing to an answer. Do not stop at the first plausible-looking option.

**found_answer field**:
- Set found_answer=true when the answer is clearly determined and supported by sources, even
  if not every source confirmed it. Consistent evidence from one or more credible sources is
  sufficient — do NOT demand unanimous confirmation before accepting an answer.
- Set found_answer=false only when the answer is genuinely uncertain, contradicted, or missing.

**answer field**:
- If found_answer is true, extract the final answer as concisely as possible.
- Preserve exact notation, qualifiers, and any symbolic form required by the task.
- **Scope the answer strictly to what the task asks for.** If the task asks for one specific
  item and additional related material was provided only as context or examples, include ONLY
  the requested item — do not append the auxiliary context to the answer.

**has_conflict field**:
- Set has_conflict=true if sources produced meaningfully contradictory conclusions on
  task-relevant points. Minor wording or coverage differences do not count as conflicts.

**Source authority for image tasks**:
- When a **[google_lens_search]** report is present, the task involves an image.
  In that case, the Google Lens result carries higher evidentiary authority than
  LLM-generated reasoning for questions about the image's origin, title, author,
  or identity — because Google Lens performs a direct reverse image search against
  a real index, while LLM sources reason from general knowledge.
- If **[google_lens_search]** returned a concrete identification and an LLM source
  returned a different or vaguer answer, prefer the Google Lens result and do NOT
  treat this as a conflict requiring another round.
- Only set has_conflict=true for a Google Lens result if it is internally
  inconsistent or clearly impossible.

- Respond in JSON with fields: reasoning, found_answer, answer, has_conflict.
</eval_rules>
"""

EVAL_SYSTEM_PROMPT_TEMPLATE = """
{{ eval_agent_profile }}
{{ eval_rules }}
"""

EVAL_AGENT_MESSAGE_TEMPLATE = """Task: {{ task }}
{% if previous %}

Prior research rounds:
{{ previous }}
{% endif %}

Current round — search reports:
{{ content }}

Based on all search reports above:
1. Detect any conflicts or contradictions between sources.
2. Write reasoning: key conclusions + critical evidence. If found_answer=true, explain why the answer is correct. If conflict, describe the exact disagreement. **If the task is multiple-choice, explicitly analyze every option (why correct or incorrect) before committing to an answer.**
3. Determine if the task is fully answered (found_answer: true/false). Trust consistent evidence from one or more credible sources — do NOT require unanimous confirmation.
4. If found_answer is true, extract the final answer exactly as needed. Scope strictly to what the task asks — do NOT include auxiliary examples or context provided only as hints.
5. Set has_conflict=true only for meaningful contradictions on task-relevant points.
6. If found_answer is false, ensure reasoning precisely describes the conflict or gap so the next round's query can target it."""

EVAL_SYSTEM_PROMPT = {
    "name": "deep_researcher_v3_eval_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for per-round research completeness evaluation",
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
        "eval_rules": {
            "name": "eval_rules",
            "type": "system_prompt",
            "description": "Rules for evaluating research completeness and detecting conflicts.",
            "require_grad": True,
            "template": None,
            "variables": EVAL_RULES,
        },
    },
}

EVAL_AGENT_MESSAGE_PROMPT = {
    "name": "deep_researcher_v3_eval_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Per-round dynamic context for research completeness evaluation",
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
        "previous": {
            "name": "previous",
            "type": "agent_message_prompt",
            "description": "Execution log of prior research rounds (reasoning + answer per round); empty string on round 1.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "content": {
            "name": "content",
            "type": "agent_message_prompt",
            "description": "All search reports from this round, each prefixed with **[source-name]** (e.g., **[firecrawl_search]**, **[model-name]**).",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
    },
}


@PROMPT.register_module(force=True)
class DeepResearcherV3EvalSystemPrompt(Prompt):
    """System prompt for per-round research completeness evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_researcher_v3_eval")
    description: str = Field(default="System prompt for per-round research completeness evaluation")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=EVAL_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepResearcherV3EvalAgentMessagePrompt(Prompt):
    """Agent message prompt for per-round research completeness evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_researcher_v3_eval")
    description: str = Field(default="Per-round dynamic context for research completeness evaluation")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=EVAL_AGENT_MESSAGE_PROMPT)
