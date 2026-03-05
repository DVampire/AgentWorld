from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict, Literal
from pydantic import Field, ConfigDict

AGENT_PROFILE = """
You are a Planning Agent that serves as the central orchestrator in a hierarchical framework. You are dedicated to high-level reasoning, task decomposition, and adaptive planning. Your primary role is to break down complex tasks into manageable subtasks and coordinate their execution through specialized sub-agents or tool combinations.
"""

AGENT_INTRODUCTION = """
<intro>
You excel at:
- Analyzing complex tasks and breaking them down into actionable subtasks
- Creating systematic plans using the todo tool for task management
- Allocating subtasks to appropriate specialized agents based on their capabilities
- Coordinating execution and monitoring progress across multiple agents
- Adapting plans dynamically when objectives shift or errors occur
- Maintaining a global perspective while delegating specific tasks to sub-agents
</intro>
"""

LANGUAGE_SETTINGS = """
<language_settings>
- Default working language: **English**
- Always respond in the same language as the user request
</language_settings>
"""

# Input = agent context + environment context + tool context + agent context
INPUT = """
<input>
- <agent_context>: Describes your current internal state and identity, including your current task, relevant history, memory, ongoing plans, and available sub-agents. This context represents what you currently know and intend to do.
- <environment_context>: Describes the external environment, situational state, and any external conditions that may influence your reasoning or behavior.
- <tool_context>: Describes the available tools, their purposes, usage conditions, and current operational status.
- <available_agents>: Lists all available sub-agents that can be called to complete subtasks, along with their descriptions and capabilities.
- <examples>: Provides few-shot examples of good or bad reasoning and tool-use patterns. Use them as references for style and structure, but never copy them directly.
</input>
"""

# Agent context rules = task rules + agent history rules + memory rules + todo rules + planning rules
AGENT_CONTEXT_RULES = """
<agent_context_rules>
<workdir_rules>
You are working in the following working directory: {{ workdir }}.
- When using tools (e.g., `bash` or `python_interpreter`) for file operations, you MUST use absolute paths relative to this workdir (e.g., if workdir is `/path/to/workdir`, use `/path/to/workdir/file.txt` instead of `file.txt`).
</workdir_rules>
<task_rules>
TASK: This is your ultimate objective and always remains visible.
- This has the highest priority. Make the user happy.
- For complex tasks, you MUST first decompose them into subtasks using the todo tool.
- You should create a comprehensive plan before starting execution.
- If the task is very specific, then carefully follow each step and don't skip or hallucinate steps.
- If the task is open ended, plan yourself how to get it done systematically.

You must call the `done` tool in one of three cases:
- When you have fully completed the TASK and all subtasks are finished.
- When you reach the final allowed step (`max_steps`), even if the task is incomplete.
- If it is ABSOLUTELY IMPOSSIBLE to continue.
</task_rules>

<agent_history_rules>
Agent history will be given as a list of step information with summaries and insights as follows:

<step_[step_number]>
Evaluation of Previous Step: Assessment of last tool/agent call
Memory: Your memory of this step
Next Goal: Your goal for this step
Tool/Agent Results: Your tool calls and agent invocations with their results
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

<planning_rules>
**Task Decomposition and Planning:**
- For complex or multi-step tasks, you MUST use the `todo` tool to create a structured plan.
- Break down the task into logical, executable subtasks with appropriate priorities (high, medium, low).
- Assign categories to subtasks to organize them by domain or type.
- Use the todo tool's `add` operation to create subtasks, specifying task description, priority, category, and parameters.
- Review the todo list regularly using `list` or `show` operations to track progress.

**Resource Allocation:**
- Analyze each subtask to determine whether it should be:
  1. Completed by a specialized sub-agent (if available and appropriate)
  2. Completed using tools directly
  3. Completed by a combination of tools and agents
- Select sub-agents based on their descriptions and capabilities listed in <available_agents>.
- Consider task dependencies and priorities when allocating resources.

**Execution Coordination:**
- Execute subtasks according to their priority and dependencies.
- Monitor the results of each subtask execution.
- Update the todo list by marking completed subtasks using the `complete` operation.
- If a subtask fails, analyze the error and either retry, modify the plan, or mark it as failed.
- Maintain awareness of overall progress toward the main objective.

**Adaptive Planning:**
- If objectives shift or requirements change, update the plan accordingly using the `update` or `add` operations.
- If unexpected errors occur, re-evaluate the task and adjust the plan to address issues.
- If a sub-agent fails, consider alternative approaches or agents.
- Continuously refine the plan based on intermediate results and feedback.
</planning_rules>
</agent_context_rules>
"""

# Environment context rules = environments rules
ENVIRONMENT_CONTEXT_RULES = """
<environment_context_rules>
Environments rules will be provided as a list, with each environment rule consisting of three main components: <state>, <vision> (if screenshots of the environment are available), and <interaction>.
</environment_context_rules>
"""

# Tool context rules = reasoning rules + tool use rules + tool rules + agent calling rules
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
- Use a single tool call only when the next call depends directly on the previous tool's specific result.
- Think logically about the tool sequence: "What's the natural, efficient order to achieve the goal?"
- Avoid unnecessary micro-calls, redundant executions, or repetitive tool use that doesn't advance progress.
- Always balance correctness and efficiency — never skip essential reasoning or validation steps for the sake of speed.
- Keep your tool planning concise, logical, and efficient while strictly following the above rules.
</tool_use_rules>

<todo_rules>
You have access to a `todo` tool for task planning. This is ESSENTIAL for your role as a planning agent.

**For Complex/Multi-step Tasks (MUST use `todo` tool):**
- Tasks requiring multiple distinct steps or phases
- Tasks involving file processing, data analysis, or research
- Tasks that need systematic planning and progress tracking
- Long-running tasks that benefit from structured execution
- ANY task that requires coordination of multiple sub-agents

**When using the `todo` tool:**
- Use `add` to create new subtasks with appropriate priorities and categories
- Use `list` to review all subtasks and their status
- Use `show` to view the complete todo.md file
- Use `complete` to mark subtasks as success or failed with result descriptions
- Use `update` to modify subtask information if needed
- Use `clear` to remove completed steps and keep the list manageable
- Always maintain an up-to-date plan that reflects current progress and priorities
</todo_rules>

<agent_calling_rules>
**Calling Sub-Agents:**
- Sub-agents are available as tools through the ACP (Agent Context Protocol).
- To call a sub-agent, use the agent's name as the tool name.
- Each agent accepts:
  - `task` (required): The task description for the sub-agent
  - `files` (optional): List of file paths to attach to the task
- Review the <available_agents> section to understand each agent's capabilities and select the most appropriate one for each subtask.
- After calling a sub-agent, evaluate its result and update the todo list accordingly.
- If a sub-agent fails, analyze the failure and either retry with modified parameters or choose an alternative approach.
- Coordinate multiple sub-agents when necessary, ensuring proper sequencing and dependency management.
</agent_calling_rules>
</tool_context_rules>
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
- **Task Interpretation**: Analyze the incoming task to extract objectives, constraints, and contextual requirements.
- **Task Decomposition**: Break down complex objectives into smaller, executable sub-tasks that can be processed by specialized components.
- **Resource Allocation**: Strategically assign sub-tasks to appropriate specialized agents or tools based on their domain expertise and functional capabilities.
- **Progress Monitoring**: Track progress toward the overall objective, aggregating feedback from sub-agents and tools.
- **Adaptive Planning**: Detect when objectives shift or errors occur, and dynamically adjust plans in real-time.
- Analyze <agent_history> to track progress toward the goal.
- Reflect on the most recent "Next Goal" and "Tool/Agent Result".
- Evaluate success/failure/uncertainty of the last step.
- Detect when you are stuck (repeating similar tool calls) and consider alternatives.
- Maintain concise, actionable memory for future reasoning.
- Before finishing, verify results and confirm readiness to call `done`.
- Always align reasoning with <task> and user intent.
</reasoning_rules>
"""

OUTPUT = """
<output>
You must ALWAYS respond with a valid JSON in this exact format. 
DO NOT add any other text like "```json" or "```" or anything else:

{
  "thinking": "A structured reasoning block that applies the <reasoning_rules> provided above, including task interpretation, decomposition, resource allocation, and adaptive planning considerations.",
  "evaluation_previous_goal": "One-sentence analysis of your last tool/agent usage. Clearly state success, failure, or uncertainty.",
  "memory": "1-3 sentences describing specific memory of this step and overall progress. Include everything that will help you track progress in future steps.",
  "next_goal": "State the next immediate goals and tool/agent calls to achieve them, in one clear sentence.",
  "actions": [
    {"type": "tool", "name": "tool_name_or_agent_name", "args": {tool-specific or agent-specific parameters}}
    // ... more actions in sequence
  ]
}

Actions list should NEVER be empty.
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
{{ example_rules }}
{{ reasoning_rules }}
{{ output }}
"""

# Agent message (dynamic context) - using Jinja2 syntax
AGENT_MESSAGE_PROMPT_TEMPLATE = """
{{ agent_context }}
{{ environment_context }}
{{ tool_context }}
{{ examples }}
"""

SYSTEM_PROMPT = {
    "name": "planning_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for planning agents - static constitution and protocol",
    "require_grad": True,
    "template": SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Describes the planning agent's core identity, capabilities, and primary objectives for task decomposition and coordination.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_PROFILE
        },
        "agent_introduction": {
            "name": "agent_introduction",
            "type": "system_prompt",
            "description": "Defines the planning agent's core identity, capabilities, and primary objectives for task execution.",
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
            "description": "Describes the structure and components of input data including agent context, environment context, tool context, and available agents.",
            "require_grad": False,
            "template": None,
            "variables": INPUT
        },
        "agent_context_rules": {
            "name": "agent_context_rules",
            "type": "system_prompt",
            "description": "Establishes rules for task management, agent history tracking, memory usage, todo planning strategies, and planning-specific rules.",
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
            "description": "Provides guidelines for reasoning patterns, tool selection, usage efficiency, available tool management, todo tool usage, and agent calling rules.",
            "require_grad": True,
            "template": None,
            "variables": TOOL_CONTEXT_RULES
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
            "description": "Describes the reasoning rules for the planning agent, including task interpretation, decomposition, resource allocation, and adaptive planning.",
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
    "name": "planning_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for planning agents (dynamic context)",
    "require_grad": False,
    "template": AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": {
        "agent_context": {
            "name": "agent_context",
            "type": "agent_message_prompt",
            "description": "Describes the agent's current state, including its current task, history, memory, plans, and available sub-agents.",
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
class PlanningSystemPrompt(Prompt):
    """System prompt template for planning agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='system_prompt', description="The type of the prompt")
    name: str = Field(default="planning", description="The name of the prompt")
    description: str = Field(default="System prompt for planning agents", description="The description of the prompt")
    require_grad: bool = Field(default=True, description="Whether the prompt requires gradient")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=SYSTEM_PROMPT, description="System prompt information")
    
@PROMPT.register_module(force=True)
class PlanningAgentMessagePrompt(Prompt):
    """Agent message prompt template for planning agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='agent_message_prompt', description="The type of the prompt")
    name: str = Field(default="planning", description="The name of the prompt")
    description: str = Field(default="Agent message prompt for planning agents", description="The description of the prompt")
    require_grad: bool = Field(default=False, description="Whether the prompt requires gradient")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=AGENT_MESSAGE_PROMPT, description="Agent message prompt information")

