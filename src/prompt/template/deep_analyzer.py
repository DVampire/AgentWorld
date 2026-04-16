from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict
from pydantic import Field, ConfigDict


# ===========================================================================
# Classify — file type classification
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

CLASSIFY_AGENT_PROFILE = """
You are an expert at classifying file types based on their content and purpose.
Classify each file as one of: 'text', 'pdf', 'image', 'audio', or 'video'.
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT_TEMPLATE = """
{{ classify_agent_profile }}
"""

CLASSIFY_AGENT_MESSAGE_TEMPLATE = """
Classify the following files by type:
- 'text': Text, markup, programming, documents (DOCX, XLSX, PPTX), or compressed files
- 'pdf': PDF files
- 'image': Image files (JPG, PNG, GIF, BMP, WebP, TIFF, SVG)
- 'audio': Audio files (MP3, WAV, OGG, FLAC, AAC, M4A)
- 'video': Video files (MP4, AVI, MOV, WMV, WebM)

IMPORTANT: Return the `file` field with the ABSOLUTE path as given below — do NOT shorten,
relativize, or modify paths in any way. Each path is an absolute file system path.

Files to classify:
{{ file_list }}
"""

# ---------------------------------------------------------------------------
# Prompt config dicts
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = {
    "name": "deep_analyzer_classify_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for file type classification",
    "require_grad": False,
    "template": CLASSIFY_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "classify_agent_profile": {
            "name": "classify_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the file classifier.",
            "require_grad": False,
            "template": None,
            "variables": CLASSIFY_AGENT_PROFILE,
        },
    },
}

CLASSIFY_AGENT_MESSAGE_PROMPT = {
    "name": "deep_analyzer_classify_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for file type classification",
    "require_grad": False,
    "template": CLASSIFY_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "file_list": {
            "name": "file_list",
            "type": "agent_message_prompt",
            "description": "Newline-separated list of file paths to classify.",
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
class DeepAnalyzerClassifySystemPrompt(Prompt):
    """System prompt for deep analyzer file classification."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_analyzer_classify")
    description: str = Field(default="System prompt for file type classification")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=CLASSIFY_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepAnalyzerClassifyAgentMessagePrompt(Prompt):
    """Agent message prompt for deep analyzer file classification."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_analyzer_classify")
    description: str = Field(default="Agent message for file type classification")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=CLASSIFY_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Chunk — markdown chunk analysis
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

CHUNK_AGENT_PROFILE = """
You are an expert at extracting key information from document chunks.
Analyze the provided text chunk and determine if it answers the given task.
"""

CHUNK_RULES = """
<chunk_rules>
- Extract only information relevant to the task.
- Provide a concise summary (2-3 sentences).
- If the chunk contains the answer, set found_answer=true and state the answer clearly.
- If not, summarize what relevant information was found.
</chunk_rules>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

CHUNK_SYSTEM_PROMPT_TEMPLATE = """
{{ chunk_agent_profile }}
{{ chunk_rules }}
"""

CHUNK_AGENT_MESSAGE_TEMPLATE = """
Task: {{ task }}

Document chunk (lines {{ start_line }}-{{ end_line }}):
{{ chunk_text }}

Analyze this chunk and extract information relevant to the task.
"""

# ---------------------------------------------------------------------------
# Prompt config dicts
# ---------------------------------------------------------------------------

CHUNK_SYSTEM_PROMPT = {
    "name": "deep_analyzer_chunk_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for markdown chunk analysis",
    "require_grad": True,
    "template": CHUNK_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "chunk_agent_profile": {
            "name": "chunk_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the chunk analyzer.",
            "require_grad": False,
            "template": None,
            "variables": CHUNK_AGENT_PROFILE,
        },
        "chunk_rules": {
            "name": "chunk_rules",
            "type": "system_prompt",
            "description": "Rules for analyzing document chunks.",
            "require_grad": True,
            "template": None,
            "variables": CHUNK_RULES,
        },
    },
}

CHUNK_AGENT_MESSAGE_PROMPT = {
    "name": "deep_analyzer_chunk_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for chunk analysis",
    "require_grad": False,
    "template": CHUNK_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The analysis task.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "start_line": {
            "name": "start_line",
            "type": "agent_message_prompt",
            "description": "First line number of the chunk.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "end_line": {
            "name": "end_line",
            "type": "agent_message_prompt",
            "description": "Last line number of the chunk.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "chunk_text": {
            "name": "chunk_text",
            "type": "agent_message_prompt",
            "description": "The text content of the chunk.",
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
class DeepAnalyzerChunkSystemPrompt(Prompt):
    """System prompt for deep analyzer chunk analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_analyzer_chunk")
    description: str = Field(default="System prompt for markdown chunk analysis")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=CHUNK_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepAnalyzerChunkAgentMessagePrompt(Prompt):
    """Agent message prompt for deep analyzer chunk analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_analyzer_chunk")
    description: str = Field(default="Dynamic context for chunk analysis")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=CHUNK_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Task — task-only analysis (no files)
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

TASK_AGENT_PROFILE = """
You are an expert at solving complex reasoning tasks, text games, math problems, and logic puzzles.
Analyze the task step by step and provide a comprehensive answer.
"""

TASK_RULES = """
<task_rules>
- Break down the task into components.
- Identify key information and constraints.
- Apply logical reasoning or mathematical operations.
- If you find the complete answer, set found_answer=true and state it clearly.
- Provide a concise summary (2-4 sentences) of your analysis.
</task_rules>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

TASK_SYSTEM_PROMPT_TEMPLATE = """
{{ task_agent_profile }}
{{ task_rules }}
"""

TASK_AGENT_MESSAGE_TEMPLATE = """
Task: {{ task }}

Round {{ round_number }} of {{ max_rounds }}.
{% if previous_summaries %}
Previous analysis rounds:
{{ previous_summaries }}
{% endif %}

Analyze the task step by step and provide your findings for this round.
"""

# ---------------------------------------------------------------------------
# Prompt config dicts
# ---------------------------------------------------------------------------

TASK_SYSTEM_PROMPT = {
    "name": "deep_analyzer_task_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for task-only analysis",
    "require_grad": True,
    "template": TASK_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "task_agent_profile": {
            "name": "task_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the task analyzer.",
            "require_grad": False,
            "template": None,
            "variables": TASK_AGENT_PROFILE,
        },
        "task_rules": {
            "name": "task_rules",
            "type": "system_prompt",
            "description": "Rules for task analysis.",
            "require_grad": True,
            "template": None,
            "variables": TASK_RULES,
        },
    },
}

TASK_AGENT_MESSAGE_PROMPT = {
    "name": "deep_analyzer_task_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for task-only analysis",
    "require_grad": False,
    "template": TASK_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The analysis task.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "round_number": {
            "name": "round_number",
            "type": "agent_message_prompt",
            "description": "Current round number.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "max_rounds": {
            "name": "max_rounds",
            "type": "agent_message_prompt",
            "description": "Maximum number of analysis rounds.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "previous_summaries": {
            "name": "previous_summaries",
            "type": "agent_message_prompt",
            "description": "Summaries from previous rounds.",
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
class DeepAnalyzerTaskSystemPrompt(Prompt):
    """System prompt for deep analyzer task-only analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_analyzer_task")
    description: str = Field(default="System prompt for task-only analysis")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=TASK_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepAnalyzerTaskAgentMessagePrompt(Prompt):
    """Agent message prompt for deep analyzer task-only analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_analyzer_task")
    description: str = Field(default="Dynamic context for task-only analysis")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=TASK_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Summarize — synthesize multiple summaries
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

SUMMARIZE_AGENT_PROFILE = """
You are an expert at synthesizing information from multiple analysis summaries.
Integrate all findings and determine if the task has been answered.
"""

SUMMARIZE_RULES = """
<summarize_rules>
- Integrate all findings into a comprehensive summary (3-5 sentences).
- Determine if the task has been fully answered based on all evidence.
- If the answer is found, set found_answer=true and provide the complete answer.
- If not, summarize what was discovered and what gaps remain.
</summarize_rules>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM_PROMPT_TEMPLATE = """
{{ summarize_agent_profile }}
{{ summarize_rules }}
"""

SUMMARIZE_AGENT_MESSAGE_TEMPLATE = """
Task: {{ task }}

Analysis summaries:
{{ summaries_text }}

Synthesize all information and determine if the task has been answered.
"""

# ---------------------------------------------------------------------------
# Prompt config dicts
# ---------------------------------------------------------------------------

SUMMARIZE_SYSTEM_PROMPT = {
    "name": "deep_analyzer_summarize_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for summarizing analysis findings",
    "require_grad": True,
    "template": SUMMARIZE_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "summarize_agent_profile": {
            "name": "summarize_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the summarizer.",
            "require_grad": False,
            "template": None,
            "variables": SUMMARIZE_AGENT_PROFILE,
        },
        "summarize_rules": {
            "name": "summarize_rules",
            "type": "system_prompt",
            "description": "Rules for synthesizing summaries.",
            "require_grad": True,
            "template": None,
            "variables": SUMMARIZE_RULES,
        },
    },
}

SUMMARIZE_AGENT_MESSAGE_PROMPT = {
    "name": "deep_analyzer_summarize_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for summary synthesis",
    "require_grad": False,
    "template": SUMMARIZE_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The analysis task.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "summaries_text": {
            "name": "summaries_text",
            "type": "agent_message_prompt",
            "description": "Bulleted list of analysis summaries.",
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
class DeepAnalyzerSummarizeSystemPrompt(Prompt):
    """System prompt for deep analyzer summary synthesis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_analyzer_summarize")
    description: str = Field(default="System prompt for summarizing analysis findings")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=SUMMARIZE_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepAnalyzerSummarizeAgentMessagePrompt(Prompt):
    """Agent message prompt for deep analyzer summary synthesis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_analyzer_summarize")
    description: str = Field(default="Dynamic context for summary synthesis")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=SUMMARIZE_AGENT_MESSAGE_PROMPT)


# ===========================================================================
# Direct — direct multimodal file analysis (image / audio / video / PDF)
# ===========================================================================

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

DIRECT_AGENT_PROFILE = """
You are an expert at analyzing multimodal content including images, audio, video, and PDF documents.
Extract key information that helps answer the given task.
"""

DIRECT_RULES = """
<direct_analysis_rules>
- Focus on information relevant to the task.
- Provide a concise summary (2-3 sentences) of findings.
- If the content contains the answer to the task, set found_answer=true and state the answer clearly.
- If not, describe what relevant information was extracted.
</direct_analysis_rules>
"""

# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------

DIRECT_SYSTEM_PROMPT_TEMPLATE = """
{{ direct_agent_profile }}
{{ direct_rules }}
"""

DIRECT_AGENT_MESSAGE_TEMPLATE = """
Task: {{ task }}

Analyze the attached {{ file_type }} and extract information relevant to the task.
If the content contains the answer, clearly state it.
"""

# ---------------------------------------------------------------------------
# Prompt config dicts
# ---------------------------------------------------------------------------

DIRECT_SYSTEM_PROMPT = {
    "name": "deep_analyzer_direct_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for direct multimodal file analysis",
    "require_grad": True,
    "template": DIRECT_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "direct_agent_profile": {
            "name": "direct_agent_profile",
            "type": "system_prompt",
            "description": "Core identity of the direct file analyzer.",
            "require_grad": False,
            "template": None,
            "variables": DIRECT_AGENT_PROFILE,
        },
        "direct_rules": {
            "name": "direct_rules",
            "type": "system_prompt",
            "description": "Rules for direct file analysis.",
            "require_grad": True,
            "template": None,
            "variables": DIRECT_RULES,
        },
    },
}

DIRECT_AGENT_MESSAGE_PROMPT = {
    "name": "deep_analyzer_direct_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Dynamic context for direct multimodal file analysis",
    "require_grad": False,
    "template": DIRECT_AGENT_MESSAGE_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "The analysis task.",
            "require_grad": False,
            "template": None,
            "variables": None,
        },
        "file_type": {
            "name": "file_type",
            "type": "agent_message_prompt",
            "description": "The type of file being analyzed (image, audio, video, PDF).",
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
class DeepAnalyzerDirectSystemPrompt(Prompt):
    """System prompt for deep analyzer direct multimodal file analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="system_prompt")
    name: str = Field(default="deep_analyzer_direct")
    description: str = Field(default="System prompt for direct multimodal file analysis")
    require_grad: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=DIRECT_SYSTEM_PROMPT)


@PROMPT.register_module(force=True)
class DeepAnalyzerDirectAgentMessagePrompt(Prompt):
    """Agent message prompt for deep analyzer direct multimodal file analysis."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default="agent_message_prompt")
    name: str = Field(default="deep_analyzer_direct")
    description: str = Field(default="Dynamic context for direct multimodal file analysis")
    require_grad: bool = Field(default=False)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    prompt_config: Dict[str, Any] = Field(default=DIRECT_AGENT_MESSAGE_PROMPT)
