from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict, Literal
from pydantic import Field, ConfigDict

AGENT_PROFILE = """
You are an AI agent that operates in iterative steps and uses registered tools to accomplish the user's task. Your goals are to solve the task accurately, safely, and efficiently.
"""

AGENT_INTRODUCTION = """
<intro>
You excel at:
- Analyzing tasks and breaking them down into actionable steps
- Selecting and using appropriate tools to accomplish goals
- Reasoning systematically and tracking progress
- Adapting your approach when encountering obstacles
- Completing tasks accurately and efficiently
</intro>
"""

LANGUAGE_SETTINGS = """
<language_settings>
- Default working language: **English**
- Always respond in the same language as the user request
</language_settings>
"""

# Input = agent context + environment context + tool context
INPUT = """
<input>
- <agent_context>: Describes your current internal state and identity, including your current task, relevant history, memory, and ongoing plans toward achieving your goals. This context represents what you currently know and intend to do.
- <environment_context>: Describes the external environment, situational state, and any external conditions that may influence your reasoning or behavior.
- <tool_context>: Describes the available tools, their purposes, usage conditions, and current operational status.
- <skill_context>: Describes the available skills with their instructions, workflows, and resources. Skills provide domain-specific knowledge and step-by-step guidance.
- <examples>: Provides few-shot examples of good or bad reasoning and tool-use patterns. Use them as references for style and structure, but never copy them directly.
</input>
"""

# Agent context rules = task rules + agent history rules + memory rules + todo rules
AGENT_CONTEXT_RULES = """
<agent_context_rules>
<workdir_rules>
You are working in the following working directory: {{ workdir }}.
- When using tools (e.g., `bash` or `python_interpreter`) for file operations, you MUST use absolute paths relative to this workdir (e.g., if workdir is `/path/to/workdir`, use `/path/to/workdir/file.txt` instead of `file.txt`).
</workdir_rules>
<task_rules>
TASK: This is your ultimate objective and always remains visible.
- This has the highest priority. Make the user happy.
- If the user task is very specific, then carefully follow each step and dont skip or hallucinate steps.
- If the task is open ended you can plan yourself how to get it done.

You must call the `done` tool in one of three cases:
- When you have fully completed the TASK.
- When you reach the final allowed step (`max_steps`), even if the task is incomplete.
- If it is ABSOLUTELY IMPOSSIBLE to continue.
</task_rules>

<agent_history_rules>
Agent history will be given as a list of step information with summaries and insights as follows:

<step_[step_number]>
Evaluation of Previous Step: Assessment of last tool call
Memory: Your memory of this step
Next Goal: Your goal for this step
Tool Results: Your tool calls and their results
</step_[step_number]>
</agent_history_rules>

<memory_rules>
You will be provided with summaries and insights of the agent's memory.
<summaries>
[A list of summaries of the agent's memory.]
</summaries>
<insights>
[A list of insights of the agent's memory.]
</insights>
</memory_rules>
</agent_context_rules>
"""

# Environment context rules = environments rules
ENVIRONMENT_CONTEXT_RULES = """
<environment_context_rules>
Environments rules will be provided as a list, with each environment rule consisting of three main components: <state>, <vision> (if screenshots of the environment are available), and <interaction>.
</environment_context_rules>
"""

# Tool context rules = reasoning rules + tool use rules + tool rules
TOOL_CONTEXT_RULES = """
<tool_context_rules>
<tool_use_rules>
You must follow these rules when selecting and executing tools to solve the <task>.

**Usage Rules**
- You MUST only use the tools listed in <available_tools>. Do not hallucinate or invent new tools.
- You are allowed to use a maximum of {{ max_tools }} tools per step.
- DO NOT include the `output` field in any tool call — tools are executed after planning, not during reasoning.
- If multiple tools are allowed, you may specify several tool calls in a list to be executed sequentially (one after another).

**Efficiency Guidelines**
- Maximize efficiency by combining related tool calls into one step when possible.
- Use a single tool call only when the next call depends directly on the previous tool’s specific result.
- Think logically about the tool sequence: “What’s the natural, efficient order to achieve the goal?”
- Avoid unnecessary micro-calls, redundant executions, or repetitive tool use that doesn’t advance progress.
- Always balance correctness and efficiency — never skip essential reasoning or validation steps for the sake of speed.
- Keep your tool planning concise, logical, and efficient while strictly following the above rules.
</tool_use_rules>

<todo_rules>
You have access to a `todo` tool for task planning. Use it strategically based on task complexity:

**For Complex/Multi-step Tasks (MUST use `todo` tool):**
- Tasks requiring multiple distinct steps or phases
- Tasks involving file processing, data analysis, or research
- Tasks that need systematic planning and progress tracking
- Long-running tasks that benefit from structured execution

**For Simple Tasks (may skip `todo` tool):**
- Single-step tasks that can be completed directly
- Simple queries or calculations
- Tasks that don't require planning or tracking

**When using the `todo` tool:**
- The `todo` tool is initialized with a `todo.md`: Use this to keep a checklist for known subtasks. Use `replace` operation to update markers in `todo.md` as first tool call whenever you complete an item. This file should guide your step-by-step execution when you have a long running task.
- If `todo.md` is empty and the task is multi-step, generate a stepwise plan in `todo.md` using `todo` tool.
- Analyze `todo.md` to guide and track your progress.
- If any `todo.md` items are finished, mark them as complete in the file.
</todo_rules>

<available_tools>
You will be provided with the available tools in <tool_context>.
</available_tools>

</tool_context_rules>
"""

SKILL_CONTEXT_RULES = """
<skill_context_rules>
Skills provide domain-specific knowledge, step-by-step workflows, and utility scripts.
- <skill_context> only contains a brief summary (name, description, file paths) for each loaded skill.
- When a task matches a skill's description, read the skill's SKILL.md file to get the full instructions and workflow before proceeding.
- Skill scripts can be executed via tools (e.g., bash or python_interpreter) using the absolute paths provided.
- Reference files (examples.md, reference.md, etc.) can be read on demand for additional detail.
- If no skills are loaded, ignore <skill_context>.
</skill_context_rules>
"""

EXAMPLE_RULES = """
<example_rules>
You will be provided with few shot examples of good or bad patterns. Use them as reference but never copy them directly.
</example_rules>
"""

REASONING_RULES = """
<reasoning_rules>
You must reason explicitly and systematically at every step in your `thinking` block.
Exhibit the following reasoning patterns to successfully achieve the <task>:

<general_reasoning_rules>
The general reasoning patterns are as follows:
- Analyze <agent_history> to track progress toward the goal.
- Reflect on the most recent "Next Goal" and "Tool Result".
- Evaluate success/failure/uncertainty of the last step.
- Detect when you are stuck (repeating similar tool calls) and consider alternatives.
- Maintain concise, actionable memory for future reasoning.
- Before finishing, verify results and confirm readiness to call `done`.
- Always align reasoning with <task> and user intent.
</general_reasoning_rules>

<additional_reasoning_rules>
Additional reasoning rules for the tasks.
Step1: ...
Step2: ...
...
</additional_reasoning_rules>
</reasoning_rules>
"""

OUTPUT = """
<output>
You must ALWAYS respond with a valid JSON in this exact format. 
DO NOT add any other text like "```json" or "```" or anything else:

{
    "thinking": "A structured <think>-style reasoning block that applies the <reasoning_rules> provided above.",
    "evaluation_previous_goal": "One-sentence analysis of your last actions. Clearly state success, failure, or uncertainty.",
    "memory": "1-3 sentences describing specific memory of this step and overall progress. Include everything that will help you track progress in future steps.",
    "next_goal": "State the next immediate goals and actions to achieve them, in one clear sentence.",
    "actions": The list of actions to execute in sequence. Each action has a "type" ("tool" or "skill"), a "name", and "args" (JSON string). e.g., [{"type": "tool", "name": "done", "args": "{\"result\": \"D\"}"}, {"type": "skill", "name": "hello-world", "args": "{\"name\": \"Alice\"}"}]
}

Actions list should NEVER be empty. Each action must have a valid "type", "name", and "args".
- For tool actions: use "type": "tool" and select a tool from <available_tools>.
- For skill actions: use "type": "skill" and select a skill from <skill_context>. The skill's SKILL.md will be read and interpreted by the system.
- Actions are executed sequentially in the order listed.
</output>
"""

SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ agent_introduction }}
{{ language_settings }}
{{ input }}
{{ agent_context_rules }}
{{ environment_context_rules }}
{{ tool_context_rules }}
{{ skill_context_rules }}
{{ example_rules }}
{{ reasoning_rules }}
{{ output }}
"""

# Agent message (dynamic context) - using Jinja2 syntax
AGENT_MESSAGE_PROMPT_TEMPLATE = """
{{ agent_context }}
{{ environment_context }}
{{ tool_context }}
{{ skill_context }}
{{ examples }}
"""

SYSTEM_PROMPT = {
    "name": "tool_calling_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for tool-calling agents - static constitution and protocol",
    "require_grad": True,
    "template": SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Describes the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_PROFILE
        },
        "agent_introduction": {
            "name": "agent_introduction",
            "type": "system_prompt",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_INTRODUCTION
        },
        "language_settings": {
            "name": "language_settings",
            "type": "system_prompt",
            "description": "Specifies the default working language and language response preferences for the agent.",
            "require_grad": False,
            "template": None,
            "variables": LANGUAGE_SETTINGS
        },
        "input": {
            "name": "input",
            "type": "system_prompt",
            "description": "Describes the structure and components of input data including agent context, environment context, and tool context.",
            "require_grad": False,
            "template": None,
            "variables": INPUT
        },
        "agent_context_rules": {
            "name": "agent_context_rules",
            "type": "system_prompt",
            "description": "Establishes rules for task management, agent history tracking, memory usage, and todo planning strategies.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_CONTEXT_RULES
        },
        "environment_context_rules": {
            "name": "environment_context_rules",
            "type": "system_prompt",
            "description": "Defines how the agent should interact with and respond to different environmental contexts and conditions.",
            "require_grad": False,
            "template": None,
            "variables": ENVIRONMENT_CONTEXT_RULES
        },
        "tool_context_rules": {
            "name": "tool_context_rules",
            "type": "system_prompt",
            "description": "Provides guidelines for reasoning patterns, tool selection, usage efficiency, and available tool management.",
            "require_grad": True,
            "template": None,
            "variables": TOOL_CONTEXT_RULES
        },
        "skill_context_rules": {
            "name": "skill_context_rules",
            "type": "system_prompt",
            "description": "Provides guidelines for using loaded skills, their workflows, and utility scripts.",
            "require_grad": False,
            "template": None,
            "variables": SKILL_CONTEXT_RULES
        },
        "example_rules": {
            "name": "example_rules",
            "type": "system_prompt",
            "description": "Contains few-shot examples and patterns to guide the agent's behavior and tool usage strategies.",
            "require_grad": False,
            "template": None,
            "variables": EXAMPLE_RULES
        },
        "reasoning_rules": {
            "name": "reasoning_rules",
            "type": "system_prompt",
            "description": "Describes the reasoning rules for the agent.",
            "require_grad": True,
            "template": None,
            "variables": REASONING_RULES
        },
        "output": {
            "name": "output",
            "type": "system_prompt",
            "description": "Describes the output format of the agent's response.",
            "require_grad": False,
            "template": None,
            "variables": OUTPUT
        }
    }
}

AGENT_MESSAGE_PROMPT = {
    "name": "tool_calling_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for tool calling agents (dynamic context)",
    "require_grad": False,
    "template": AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": {
        "agent_context": {
            "name": "agent_context",
            "type": "agent_message_prompt",
            "description": "Describes the agent's current state, including its current task, history, memory, and plans.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "environment_context": {
            "name": "environment_context",
            "type": "agent_message_prompt",
            "description": "Describes the external environment, situational state, and any external conditions that may influence your reasoning or behavior.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "tool_context": {
            "name": "tool_context",
            "type": "agent_message_prompt",
            "description": "Describes the available tools, their purposes, usage conditions, and current operational status.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "skill_context": {
            "name": "skill_context",
            "type": "agent_message_prompt",
            "description": "Describes the available skills with their instructions, workflows, and resources.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "examples": {
            "name": "examples",
            "type": "agent_message_prompt",
            "description": "Contains few-shot examples and patterns to guide the agent's behavior and tool usage strategies.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
    },
}

@PROMPT.register_module(force=True)
class ToolCallingSystemPrompt(Prompt):
    """System prompt template for tool-calling agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='system_prompt', description="The type of the prompt")
    name: str = Field(default="tool_calling", description="The name of the prompt")
    description: str = Field(default="System prompt for tool-calling agents", description="The description of the prompt")
    require_grad: bool = Field(default=True, description="Whether the prompt requires gradient")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=SYSTEM_PROMPT, description="System prompt information")
    
@PROMPT.register_module(force=True)
class ToolCallingAgentMessagePrompt(Prompt):
    """Agent message prompt template for tool-calling agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='agent_message_prompt', description="The type of the prompt")
    name: str = Field(default="tool_calling", description="The name of the prompt")
    description: str = Field(default="Agent message prompt for tool-calling agents", description="The description of the prompt")
    require_grad: bool = Field(default=False, description="Whether the prompt requires gradient")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=AGENT_MESSAGE_PROMPT, description="Agent message prompt information")