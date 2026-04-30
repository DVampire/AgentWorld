"""Prompt templates for DeepAnalyzerV3Agent.

Five prompts cover the ThinkOutput-driven pipeline:

  | Prompt                     | 用途                                                              |
  |----------------------------|-------------------------------------------------------------------|
  | deep_analyzer_v3           | 主循环：tool_calling 风格，LLM 通过 ThinkOutput 选工具            |
  | deep_analyzer_v3_analyze   | 多模型并行分析（有文件时感知媒体，无文件直接推理）                |
  | deep_analyzer_v3_synth     | 共用汇总：保留各模型结论并整合（analyze 和 verify 均使用）        |
  | deep_analyzer_v3_eval      | 读 SynthOutput，判断 found_answer / has_conflict → EvalOutput     |
  | deep_analyzer_v3_verify    | 多模型并行反证式验证                                              |

deep_analyzer_v3 的 system prompt 结构与 tool_calling 完全对齐：
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
# Main — system prompt (tool_calling style, 4 analysis tools)
# ===========================================================================

AGENT_PROFILE = """
You are a rigorous multi-round analysis agent. You work in iterative steps and use
four internal tools to analyze a task, judge your findings, verify the answer
adversarially, and finish when the answer is confirmed.
"""

AGENT_INTRODUCTION = """
<intro>
You excel at:
- Breaking complex analysis tasks into focused investigation rounds
- Running multiple LLMs in parallel and synthesizing their findings
- Detecting conflicts between model outputs and resolving them systematically
- Adversarially verifying candidate answers before accepting them
- Completing analysis accurately and efficiently within the step budget
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
- <tool_context>: The four analysis tools available to you, with their arguments and usage rules.
</input>
"""

AGENT_CONTEXT_RULES = """
<agent_context_rules>
<task_rules>
TASK: This is your ultimate objective and always remains visible.
- Analyze the task thoroughly before committing to a course of action.
- You must call the `finish` tool in one of three cases:
  - When eval confirms found_answer=true AND at least one verify step has been run.
  - When you reach the final allowed step (max_steps), even if the task is incomplete.
  - If it is ABSOLUTELY IMPOSSIBLE to continue (e.g. all tools consistently fail).
</task_rules>

<agent_history_rules>
Your full execution history is provided in <agent_history> as a markdown analysis log.
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

**Decision rules — read the analysis.md history, then pick:**

- No history (very first step) → call `plan` to initialize the todo list and flowchart with the full analysis strategy.
- After `plan` (first time) → call `analyze` to begin the first analysis round.
- After `analyze` or `verify` → call `eval` immediately to judge the synthesis.
- After `eval`:
  - found_answer=false, has_conflict=true → call `analyze` targeting the specific conflict.
  - found_answer=false, has_conflict=false → call `analyze` from a different angle.
  - found_answer=true, no prior `verify` → call `verify` to adversarially challenge the answer.
  - found_answer=true, at least one prior `verify` → call `finish`.
  - found_answer=false, at least one prior `verify` → call `plan` to replan, then `analyze`.
- Loop guard: same conflict across 3+ consecutive analyze→eval cycles → call `finish` with best available answer.
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
- For `analyze` and `verify`, write concrete, targeted guidance — not generic instructions.
- Before calling `finish`, confirm: eval has returned found_answer=true AND a verify step exists.
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
            "name": "<plan|analyze|eval|verify|finish>",
            "args": "{}"
        }
    ]
}

- actions must contain exactly ONE action.
- For `analyze`, `eval`, `verify`: args must be "{}".
- For `plan`: args must be "{\"steps\": [\"...\", \"...\"], \"reasoning\": \"...\"}".
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
    "name": "deep_analyzer_v3_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for DeepAnalyzerV3 ThinkOutput loop",
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
#   tool_context  — available_tools block (overridden in DeepAnalyzerV3Agent)
# ===========================================================================

AGENT_MESSAGE_PROMPT_TEMPLATE = """
{{ agent_context }}
{{ tool_context }}
"""

AGENT_MESSAGE_PROMPT = {
    "name": "deep_analyzer_v3_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for DeepAnalyzerV3 ThinkOutput loop",
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
            "description": "The 4 analysis tools injected by _get_tool_context.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
    },
}


@PROMPT.register_module(force=True)
class DeepAnalyzerV3SystemPrompt(Prompt):
    """System prompt for DeepAnalyzerV3 ThinkOutput loop."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_analyzer_v3")
    description: str = Field(default="System prompt for DeepAnalyzerV3 ThinkOutput loop")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepAnalyzerV3AgentMessagePrompt(Prompt):
    """Agent message prompt for DeepAnalyzerV3 ThinkOutput loop."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_analyzer_v3")
    description: str = Field(default="Dynamic context for DeepAnalyzerV3 ThinkOutput loop")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Analyze — multi-model parallel analysis
#
# Variables:
#   task      — the analysis task
#   file_type — "image" / "audio" / "video" / "pdf" / "text" / "" (no file)
#   guidance  — analysis angles from ThinkOutput
#   synth     — previous SynthOutput text (empty on first round)
# ===========================================================================

ANALYZE_AGENT_PROFILE = """
You are a thorough analysis and reasoning assistant. Depending on the task, you may
be given a media file to examine (image, audio, video, PDF) or asked to reason directly
from your knowledge. In either case, produce a comprehensive, precise answer.
"""

ANALYZE_RULES = """
<analyze_rules>
- Answer the task directly and comprehensively based on what you observe or know.
- Follow the guidance provided — it targets the angles most important for this round.
- Take into account prior reasoning from the synthesis history — do not repeat already-settled points.
- If the task cannot be answered with confidence, clearly state what is uncertain and why.

**When a media file is provided**
- Base your answer strictly on what you can directly perceive in the file.
- If any part is ambiguous or unclear, state that explicitly rather than guessing.
- For visual content (diagrams, equations, text), transcribe verbatim rather than paraphrasing.

**Precision self-check** — before finalising any answer:
- **Counter-example**: is there any case where your conclusion breaks down?
- **Every constant and coefficient**: verify each factor, sign, and exponent from first principles.
- **Exact qualifiers**: preserve every qualifier (strict/non-strict, open/closed, superscripts).
- **Implicit conventions**: resolve domain-specific definitions before computing.
- **Multiple choice**: explicitly evaluate every option before committing.
</analyze_rules>
"""

ANALYZE_SYSTEM_PROMPT_TEMPLATE = """
{{ analyze_agent_profile }}
{{ analyze_rules }}
"""

ANALYZE_AGENT_MESSAGE_TEMPLATE = """Task: {{ task }}
{% if data.history %}

Prior analysis history (do not repeat settled points):
{% for entry in data.history %}
--- Round {{ loop.index }} ---
{{ entry }}
{% endfor %}
{% endif %}
{% if file_type %}
Analyze the provided {{ file_type }} and answer the task.
{% else %}
Provide a thorough analysis and answer the task.
{% endif %}"""

ANALYZE_SYSTEM_PROMPT = {
    "name": "deep_analyzer_v3_analyze_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for multi-model parallel analysis",
    "require_grad": True,
    "template": ANALYZE_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "analyze_agent_profile": {
            "name": "analyze_agent_profile", "type": "system_prompt",
            "description": "Core identity of the analysis component.",
            "require_grad": False, "template": None, "variables": ANALYZE_AGENT_PROFILE,
        },
        "analyze_rules": {
            "name": "analyze_rules", "type": "system_prompt",
            "description": "Rules for analysis covering both media and direct reasoning.",
            "require_grad": True, "template": None, "variables": ANALYZE_RULES,
        },
    },
}

ANALYZE_AGENT_MESSAGE_PROMPT = {
    "name": "deep_analyzer_v3_analyze_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for multi-model parallel analysis",
    "require_grad": False,
    "template": ANALYZE_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task":      {"name": "task",      "type": "agent_message_prompt", "description": "The analysis task.",                              "require_grad": False, "template": None, "variables": None},
        "file_type": {"name": "file_type", "type": "agent_message_prompt", "description": "Media type; empty for direct reasoning.",          "require_grad": False, "template": None, "variables": None},
        "data":      {"name": "data",      "type": "agent_message_prompt", "description": "Dict with key 'history': list of prior synth outputs.", "require_grad": False, "template": None, "variables": None},
    },
}


@PROMPT.register_module(force=True)
class DeepAnalyzerV3AnalyzeSystemPrompt(Prompt):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_analyzer_v3_analyze")
    description: str = Field(default="System prompt for multi-model parallel analysis")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=ANALYZE_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepAnalyzerV3AnalyzeAgentMessagePrompt(Prompt):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_analyzer_v3_analyze")
    description: str = Field(default="Dynamic context for multi-model parallel analysis")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=ANALYZE_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Synth — shared summarizer for analyze and verify parallel outputs
#
# Variables:
#   task        — the analysis task (for context)
#   raw_outputs — combined per-model outputs, each prefixed **[model-name]**
#
# Returns SynthOutput: { per_model_summaries, combined_reasoning }
# ===========================================================================

SYNTH_AGENT_PROFILE = """
You are a synthesis specialist. Multiple models have independently worked on a task.
Your job is to produce a structured summary that preserves each model's distinct conclusions
and then integrates them — without flattening disagreements into false consensus.
"""

SYNTH_RULES = """
<synth_rules>
- Each model's output is prefixed with **[model-name]**.

**per_model_summaries field**:
- One line per model: "[model-name]: <2-3 sentence summary of that model's key conclusion>"
- Preserve each model's specific claims, numbers, and qualifiers — do not paraphrase away precision.

**combined_reasoning field**:
- Identify points where all models agree — state the shared conclusion clearly.
- Identify points where models disagree — quote the specific divergence; do NOT pick a winner.
- Note any model that found something the others missed.
- Keep the integration factual and neutral; the eval step will make judgments.
</synth_rules>
"""

SYNTH_SYSTEM_PROMPT_TEMPLATE = """
{{ synth_agent_profile }}
{{ synth_rules }}
"""

SYNTH_AGENT_MESSAGE_TEMPLATE = """Task: {{ task }}

Model outputs:
{{ data }}

Produce the synthesis."""

SYNTH_SYSTEM_PROMPT = {
    "name": "deep_analyzer_v3_synth_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for shared synth (analyze and verify)",
    "require_grad": True,
    "template": SYNTH_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "synth_agent_profile": {"name": "synth_agent_profile", "type": "system_prompt", "description": "Core identity of the synthesis specialist.", "require_grad": False, "template": None, "variables": SYNTH_AGENT_PROFILE},
        "synth_rules":         {"name": "synth_rules",         "type": "system_prompt", "description": "Rules for producing a structured multi-model synthesis.", "require_grad": True, "template": None, "variables": SYNTH_RULES},
    },
}

SYNTH_AGENT_MESSAGE_PROMPT = {
    "name": "deep_analyzer_v3_synth_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for shared synth",
    "require_grad": False,
    "template": SYNTH_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {"name": "task", "type": "agent_message_prompt", "description": "The analysis task.",                                         "require_grad": False, "template": None, "variables": None},
        "data": {"name": "data", "type": "agent_message_prompt", "description": "Combined per-model outputs string, each prefixed **[model-name]**.", "require_grad": False, "template": None, "variables": None},
    },
}


@PROMPT.register_module(force=True)
class DeepAnalyzerV3SynthSystemPrompt(Prompt):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_analyzer_v3_synth")
    description: str = Field(default="System prompt for shared synth (analyze and verify)")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=SYNTH_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepAnalyzerV3SynthAgentMessagePrompt(Prompt):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_analyzer_v3_synth")
    description: str = Field(default="Dynamic context for shared synth")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=SYNTH_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Eval — read synth output, judge found_answer and has_conflict
#
# Variables:
#   task    — the analysis task
#   content — latest SynthOutput text (per_model_summaries + combined_reasoning)
#
# Returns EvalOutput: { candidate_answer, found_answer, has_conflict, conflict_description }
# ===========================================================================

EVAL_AGENT_PROFILE = """
You are a rigorous analysis evaluator. You receive a structured synthesis of multiple
model outputs and must judge: has the task been answered, and do models conflict?
"""

EVAL_RULES = """
<eval_rules>
**found_answer field**:
- True when the synthesis shows a clear, consistent answer with no unresolved conflict.
- False when the answer is uncertain, contradicted, or missing.
- Trust a consistent well-reasoned answer even without full derivation.

**candidate_answer field**:
- If found_answer=True: extract the final answer concisely, preserving exact notation.
- Scope strictly to what the task asks — omit auxiliary context or examples.
- None if found_answer=False.

**has_conflict field**:
- True only for meaningful contradictions on task-relevant points (not minor wording differences).

**conflict_description field**:
- When has_conflict=True: quote the exact disagreement and state what evidence would resolve it.
- None when has_conflict=False.

For multiple-choice tasks: verify the synthesis explicitly analyzed every option before setting found_answer=True.
</eval_rules>
"""

EVAL_SYSTEM_PROMPT_TEMPLATE = """
{{ eval_agent_profile }}
{{ eval_rules }}
"""

EVAL_AGENT_MESSAGE_TEMPLATE = """Task: {{ task }}

Analysis history:
{% for entry in data.history %}
--- Round {{ loop.index }} ---
{{ entry }}
{% endfor %}

Judge the above:
1. Set found_answer=true if a clear consistent answer exists; false otherwise.
2. If found_answer=true, extract the candidate_answer exactly as needed.
3. Set has_conflict=true only for meaningful contradictions; describe the conflict precisely.
"""

EVAL_SYSTEM_PROMPT = {
    "name": "deep_analyzer_v3_eval_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for eval tool",
    "require_grad": True,
    "template": EVAL_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "eval_agent_profile": {"name": "eval_agent_profile", "type": "system_prompt", "description": "Core identity of the eval component.",          "require_grad": False, "template": None, "variables": EVAL_AGENT_PROFILE},
        "eval_rules":         {"name": "eval_rules",         "type": "system_prompt", "description": "Rules for evaluating analysis completeness.", "require_grad": True,  "template": None, "variables": EVAL_RULES},
    },
}

EVAL_AGENT_MESSAGE_PROMPT = {
    "name": "deep_analyzer_v3_eval_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for eval tool",
    "require_grad": False,
    "template": EVAL_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {"name": "task", "type": "agent_message_prompt", "description": "The analysis task.",                                "require_grad": False, "template": None, "variables": None},
        "data": {"name": "data", "type": "agent_message_prompt", "description": "Dict with key 'history': list of prior synth outputs.", "require_grad": False, "template": None, "variables": None},
    },
}


@PROMPT.register_module(force=True)
class DeepAnalyzerV3EvalSystemPrompt(Prompt):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_analyzer_v3_eval")
    description: str = Field(default="System prompt for eval tool")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=EVAL_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepAnalyzerV3EvalAgentMessagePrompt(Prompt):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_analyzer_v3_eval")
    description: str = Field(default="Dynamic context for eval tool")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=EVAL_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Verify — adversarial multi-model verification
#
# Variables:
#   task      — the analysis task
#   file_type — media type when a file is attached; empty for direct reasoning
#   reasoning — synthesized reasoning that produced the candidate answer
#   answer    — the candidate answer to challenge
#   guidance  — specific aspects to probe (from ThinkOutput)
#
# Each model independently tries to disprove the answer.
# Results are then synthesized by deep_analyzer_v3_synth.
# ===========================================================================

VERIFY_AGENT_PROFILE = """
You are an adversarial verifier. Your job is to try to disprove the candidate answer —
probe its weaknesses, challenge every assumption, look for counter-examples and logical
gaps. Only confirm the answer if it withstands your full scrutiny.
"""

VERIFY_RULES = """
<verify_rules>
**Adversarial checklist — attempt each before concluding:**
- **Counter-example**: construct a specific case where the answer is wrong.
- **Hidden assumption**: identify any unstated assumption the reasoning relies on.
- **Edge / boundary case**: does the answer hold at the extremes of the domain?
- **Alternative interpretation**: is there a reading of the task where the answer fails?
- **Arithmetic / notation check**: re-derive key steps independently if numbers or formulas involved.
- **Scope check**: does the answer address exactly what the task asked?

Document every attack you attempted and the outcome. Be explicit: "I tried X and it held / failed because Y."
State your verdict: does the answer stand or fall?
</verify_rules>
"""

VERIFY_SYSTEM_PROMPT_TEMPLATE = """
{{ verify_agent_profile }}
{{ verify_rules }}
"""

VERIFY_AGENT_MESSAGE_TEMPLATE = """Task: {{ task }}

Analysis history:
{% for entry in data.history %}
--- Round {{ loop.index }} ---
{{ entry }}
{% endfor %}

Candidate answer: {{ data.candidate_answer }}
{% if file_type %}
Examine the provided {{ file_type }} as part of your adversarial challenge — verify that the candidate answer holds against the actual file content.
{% endif %}
Apply your adversarial checklist. Document every attack and state your verdict."""

VERIFY_SYSTEM_PROMPT = {
    "name": "deep_analyzer_v3_verify_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for adversarial verification (per-model)",
    "require_grad": True,
    "template": VERIFY_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "verify_agent_profile": {"name": "verify_agent_profile", "type": "system_prompt", "description": "Core identity of the adversarial verifier.", "require_grad": False, "template": None, "variables": VERIFY_AGENT_PROFILE},
        "verify_rules":         {"name": "verify_rules",         "type": "system_prompt", "description": "Rules for adversarial verification.",        "require_grad": True,  "template": None, "variables": VERIFY_RULES},
    },
}

VERIFY_AGENT_MESSAGE_PROMPT = {
    "name": "deep_analyzer_v3_verify_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for adversarial verification (per-model)",
    "require_grad": False,
    "template": VERIFY_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task":      {"name": "task",      "type": "agent_message_prompt", "description": "The analysis task.",                                                      "require_grad": False, "template": None, "variables": None},
        "file_type": {"name": "file_type", "type": "agent_message_prompt", "description": "Media type; empty for direct reasoning.",                              "require_grad": False, "template": None, "variables": None},
        "data":      {"name": "data",      "type": "agent_message_prompt", "description": "Dict with keys 'history' (list of synth outputs) and 'candidate_answer'.", "require_grad": False, "template": None, "variables": None},
    },
}


@PROMPT.register_module(force=True)
class DeepAnalyzerV3VerifySystemPrompt(Prompt):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_analyzer_v3_verify")
    description: str = Field(default="System prompt for adversarial verification (per-model)")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=VERIFY_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepAnalyzerV3VerifyAgentMessagePrompt(Prompt):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_analyzer_v3_verify")
    description: str = Field(default="Dynamic context for adversarial verification (per-model)")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    prompt_config: Dict[str, Any] = Field(default=VERIFY_AGENT_MESSAGE_PROMPT)
