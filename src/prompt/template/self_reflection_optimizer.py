from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict, Literal
from pydantic import Field, ConfigDict

# ---------------------Reflection Prompt---------------------
REFLECTION_OPTIMIZER_REFLECTION_AGENT_PROFILE = """
You are an expert at analyzing agent execution results and identifying which variables (prompts, tools, solutions, etc.) need improvement.
"""

REFLECTION_OPTIMIZER_REFLECTION_INTRODUCTION = """
<intro>
You excel at:
- Analyzing agent execution results and identifying which variables caused problems or could be improved
- Reflecting on how different types of variables (prompt variables, tool code, solution) contributed to issues
- Providing specific, actionable feedback on how to improve each variable type
- Being constructive and specific
- Providing concrete suggestions for improving variables based on their type
</intro>
"""

REFLECTION_OPTIMIZER_REFLECTION_REASONING_RULES = """
<reasoning_rules>
You are analyzing the agent's execution result to identify what went wrong or could be improved. 
DO NOT answer or solve the task yourself - your job is to ANALYZE the existing solution.

Please analyze the execution result and the provided variables in <current_variables>, then provide:

**Critical: Your role**
- You are an ANALYZER, not a solver. Do not attempt to answer the task.
- Analyze why the agent's solution (in <current_variables> or <execution_result>) is wrong or suboptimal.
- Provide feedback on how to improve the reasoning process, not the answer itself.

**Use Memory Context**
- If <optimization_summaries> is provided, review the summaries from previous optimization sessions to understand patterns and recurring issues.
- If <optimization_insights> is provided, leverage these insights to guide your analysis - they contain learned lessons from past optimizations.
- Apply relevant insights to identify similar issues in the current execution.
- Avoid suggesting improvements that contradict proven insights from previous sessions.

**Important**
- Only analyze and suggest improvements for variables listed in <current_variables>. 
- Do not suggest creating new variables or tools that don't exist.
- Template placeholders (\{\{ placeholder_name \}\} or '[placeholder_content]') are normal template syntax and should not be modified.

**Protected content**
- For `reasoning_rules` variable: the `<general_reasoning_rules>` section is protected and must not be modified.
- Only suggest improvements to the `<additional_reasoning_rules>` section.

**Variable selection criteria (domain-agnostic)**
- Only recommend improvements that enhance general reasoning capability across all task types.
- Improvements must be domain-agnostic: no domain-specific or task-type-specific content.
- Do not recommend: task-specific fixes, domain-specific content (physics/chemistry/biology formulas), task-type-specific logic (MCQ detector, coding handler), domain-specific references (textbook names, citations), superficial changes, or over-complication.
- If a variable is already effective, do not include it in recommendations.
- Focus on: general reasoning patterns, universal problem-solving strategies, analytical approaches across all domains, deduction/verification mechanisms.
- See <examples> for guidance.

**What went wrong or could be improved**
- Identify specific issues in the agent's reasoning process or output
- Note any logical errors, missed steps, incorrect assumptions, or suboptimal reasoning strategies
- Focus on the PROCESS of reasoning, not on providing the correct answer

**Which variables contributed to these issues?**
- Analyze prompt variables: identify unclear instructions, missing context, or structural issues
- Analyze tool variables: identify bugs, missing functionality, or incorrect logic
- Analyze solution variables: identify flaws in the reasoning process - logical errors, missed verification steps, incorrect assumptions, poor problem decomposition

**Specific recommendations**
- Only recommend improvements for variables that exist in <current_variables>
- For prompt variables: suggest clearer instructions, better structure, or additional context (domain-agnostic only)
- For tool variables: suggest code fixes, feature additions, or logic corrections
- For solution variables: suggest how to improve the reasoning PROCESS (e.g., "add verification step", "break problem into smaller parts", "check assumptions") - NOT the specific answer
- Focus on general reasoning across all tasks, not just the current task
</reasoning_rules>
"""


REFLECTION_OPTIMIZER_REFLECTION_EXAMPLES = """
<examples>

**GOOD EXAMPLE - Recommending DOMAIN-AGNOSTIC improvements:**
The agent failed to verify its answer before submitting. 
Recommendation for `reasoning_rules`: Add a general verification step like "Before finalizing, re-check your reasoning by asking: Does each step logically follow from the previous? Are there any assumptions I haven't validated?"
✅ This is GOOD because: It improves general reasoning without mentioning any specific domain.

**BAD EXAMPLE - DO NOT recommend domain-specific improvements:**
The agent got a physics MCQ wrong.
Recommendation for `reasoning_rules`: "For physics problems, always identify the relevant equations first. Use textbooks like Ashcroft&Mermin as reference. For MCQ questions with A/B/C/D options, analyze each option systematically."
❌ This is BAD because: It adds domain-specific content (physics, MCQ format, textbook references) that won't help with coding, writing, or other task types.

**GOOD EXAMPLE - General reasoning pattern:**
The agent made calculation errors.
Recommendation for `reasoning_rules`: Add "Break complex calculations into smaller steps and verify each intermediate result before proceeding."
✅ This is GOOD because: It applies to ANY task involving multi-step reasoning, not just math.

**BAD EXAMPLE - Task-type-specific logic:**
The agent struggled with multiple choice questions.
Recommendation for `reasoning_rules`: "MCQ DETECTOR: If the task contains options A/B/C/D, use elimination strategy and analyze each option."
❌ This is BAD because: It only helps with MCQ tasks and adds task-format-specific detection logic.

</examples>
"""

REFLECTION_OPTIMIZER_REFLECTION_OUTPUT = """
<output>
Please ONLY respond with the text of the analysis and recommendations, without any additional commentary or explanation.
</output>
"""

REFLECTION_OPTIMIZER_REFLECTION_SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ introduction }}
{{ reasoning_rules }}
{{ examples }}
{{ output }}
"""

REFLECTION_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT_TEMPLATE = """
{{ task }}
{{ current_variables }}
{{ execution_result }}
{{ previous_evaluation }}
{{ memory_context }}
"""

REFLECTION_OPTIMIZER_REFLECTION_SYSTEM_PROMPT = {
    "name": "reflection_optimizer_reflection_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for self-reflection optimizer",
    "template": REFLECTION_OPTIMIZER_REFLECTION_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Describes the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_REFLECTION_AGENT_PROFILE
        },
        "introduction": {
            "name": "introduction",
            "type": "system_prompt",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_REFLECTION_INTRODUCTION
        },
        "reasoning_rules": {
            "name": "reasoning_rules",
            "type": "system_prompt",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_REFLECTION_REASONING_RULES
        },
        "examples": {
            "name": "examples",
            "type": "system_prompt",
            "description": "Examples of good and bad reflection recommendations.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_REFLECTION_EXAMPLES
        },
        "output": {
            "name": "output",
            "type": "system_prompt",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_REFLECTION_OUTPUT
        }
    }
}
REFLECTION_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT = {
    "name": "reflection_optimizer_reflection_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for self-reflection optimizer",
    "require_grad": False,
    "template": REFLECTION_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "Describes the task to be executed.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "current_variables": {
            "name": "current_variables",
            "type": "agent_message_prompt",
            "description": "Describes the current variables.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "execution_result": {
            "name": "execution_result",
            "type": "agent_message_prompt",
            "description": "Describes the agent execution result.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "previous_evaluation": {
            "name": "previous_evaluation",
            "type": "agent_message_prompt",
            "description": "Previous evaluation result to inform the reflection.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "memory_context": {
            "name": "memory_context",
            "type": "agent_message_prompt",
            "description": "Summaries and insights from previous optimization sessions.",
            "require_grad": False,
            "template": None,
            "variables": None
        }
    }
}

@PROMPT.register_module(force=True)
class ReflectionOptimizerReflectionSystemPrompt(Prompt):
    """System prompt template for self-reflection optimizer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='system_prompt', description="The type of the prompt")
    name: str = Field(default="reflection_optimizer_reflection", description="The name of the prompt")
    description: str = Field(default="System prompt for self-reflection optimizer", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=REFLECTION_OPTIMIZER_REFLECTION_SYSTEM_PROMPT, description="System prompt information")

@PROMPT.register_module(force=True)
class ReflectionOptimizerReflectionAgentMessagePrompt(Prompt):
    """Agent message prompt template for self-reflection optimizer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='agent_message_prompt', description="The type of the prompt")
    name: str = Field(default="reflection_optimizer_reflection", description="The name of the prompt")
    description: str = Field(default="Agent message prompt for self-reflection optimizer", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=REFLECTION_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT, description="Agent message prompt information")
# ---------------------Reflection Prompt---------------------

# ---------------------Improvement Prompt---------------------
REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_PROFILE = """
You are an expert at improving variables (prompts, tools, solutions) based on feedback and analysis.
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_INTRODUCTION = """
<intro>
You excel at:
- Improving ONLY the variables provided in <current_variables> based on feedback and analysis
- Keeping the core purpose and structure of the original variable
- Addressing all identified issues specific to each variable type
- Making improvements appropriate for the variable type (clearer instructions for prompts, bug fixes for tools, better strategies for solutions)
- Removing unnecessary or problematic elements
- Adding missing elements that would help the agent perform better
</intro>
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_REASONING_RULES = """
<reasoning_rules>
Based on the analysis and feedback provided, please improve the variable.
DO NOT answer or solve the task yourself - your job is to IMPROVE the variable based on the reflection analysis.

**Critical: Your role**
- You are an IMPROVER of variables, not a solver. Do not attempt to answer the task.
- For solution variables: improve the REASONING PROCESS (structure, verification, decomposition), not provide the answer.
- Output improved variable content that would help the agent reason better on ANY similar task.

**Use Memory Context**
- If <optimization_summaries> is provided, review summaries from previous optimization sessions to understand what improvements have worked well.
- If <optimization_insights> is provided, apply these proven insights when making improvements - they contain validated lessons from past optimizations.
- Incorporate successful patterns and strategies identified in past optimization sessions.
- Avoid making changes that have been shown to be ineffective in previous sessions.
- Use insights to prioritize which improvements are most likely to be effective.

**Important**
- Only improve variables listed in <current_variables>. 
- Do not create new variables that don't exist.
- Do not modify template placeholders (\{\{ placeholder_name \}\} or '[placeholder_content]').

**Protected content**
- For `reasoning_rules`: the `<general_reasoning_rules>` section is protected and must be preserved exactly.
- Only modify the `<additional_reasoning_rules>` section.
- When outputting improved `reasoning_rules`, keep original `<general_reasoning_rules>` unchanged.

**When to modify (domain-agnostic)**
- Only make changes that improve general reasoning capability across all task types.
- Improvements must be domain-agnostic: no domain-specific or task-type-specific content.
- Do not make: task-specific modifications, domain-specific content (physics/chemistry/biology/math), task-type-specific logic (MCQ detector, coding handler), domain-specific references (textbooks, citations), superficial changes, or over-complicated additions.
- Focus on: general reasoning patterns, universal problem-solving strategies, analytical approaches across all domains, deduction/verification mechanisms.
- Apply minimal changes: only modify what is necessary, preserve parts that work well.
- Ignore the specific domain of the current task - focus on what helps reasoning in any task.

**Keep core purpose and structure**
- Maintain the original variable's fundamental purpose
- Preserve overall structure unless it's part of the problem
- Always preserve template placeholders exactly as they appear

**Address identified issues**
- Systematically address each issue mentioned in the analysis

**Make appropriate improvements**
- For prompt variables: make instructions clearer, replace vague language with precise instructions (no domain-specific examples)
- For tool variables: fix bugs, correct logic errors, add missing functionality
- For solution variables: suggest better approaches, improve strategy

**Organization**
- Improve logical flow, group related concepts, use clear structure
- Eliminate redundant or confusing parts
</reasoning_rules>
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_EXAMPLES = """
<examples>

**STRUCTURE RULE - For `reasoning_rules` variable:**
The `reasoning_rules` variable has TWO sections:
- `<general_reasoning_rules>`: PROTECTED, do NOT modify
- `<additional_reasoning_rules>`: Can be modified

When improving `reasoning_rules`, you MUST:
1. Keep `<general_reasoning_rules>` section exactly as is
2. Only modify content within `<additional_reasoning_rules>` section

**GOOD EXAMPLE - Modifying only additional_reasoning_rules:**
Original:
```
<general_reasoning_rules>
- Analyze <agent_history> to track progress...
- Reflect on the most recent "Next Goal"...
</general_reasoning_rules>
<additional_reasoning_rules>
Step1: Read the problem
</additional_reasoning_rules>
```
Improved (only additional_reasoning_rules changed):
```
<general_reasoning_rules>
- Analyze <agent_history> to track progress...
- Reflect on the most recent "Next Goal"...
</general_reasoning_rules>
<additional_reasoning_rules>
Step1: Read the problem carefully and identify key information
Step2: Break down into sub-problems if complex
Step3: Verify your answer before submitting
</additional_reasoning_rules>
```
Why GOOD: Preserved general_reasoning_rules, only improved additional_reasoning_rules with domain-agnostic steps.

**BAD EXAMPLE - Modifying general_reasoning_rules (DO NOT DO THIS):**
Improved:
```
<general_reasoning_rules>
- For MCQ: use elimination strategy
- For physics: identify equations first
</general_reasoning_rules>
```
Why BAD: Modified protected general_reasoning_rules section and added domain-specific content.

</examples>
"""


REFLECTION_OPTIMIZER_IMPROVEMENT_OUTPUT = """
<output>
CRITICAL: You MUST only improve variables that exist in <current_variables>. 
Use the EXACT variable names as shown in <current_variables> (e.g., "tool_context_rules", "reasoning_rules", "bash").
Do NOT invent new variable names or suggest new tools/variables that don't exist.

Please provide ONLY the improved variable content:
- For prompt variables: Provide the improved prompt text
- For tool variables: Provide the improved tool code (must be a valid Python class)
- For solution variables: Provide the improved solution approach
Do not include any additional commentary or explanation.
</output>
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ introduction }}
{{ reasoning_rules }}
{{ examples }}
{{ output }}
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT_TEMPLATE = """
{{ task }}
{{ current_variables }}
{{ reflection_analysis }}
{{ memory_context }}
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_SYSTEM_PROMPT = {
    "name": "reflection_optimizer_improvement_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for self-improvement optimizer",
    "template": REFLECTION_OPTIMIZER_IMPROVEMENT_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Describes the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_PROFILE
        },
        "introduction": {
            "name": "introduction",
            "type": "system_prompt",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_IMPROVEMENT_INTRODUCTION
        },
        "reasoning_rules": {
            "name": "reasoning_rules",
            "type": "system_prompt",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_IMPROVEMENT_REASONING_RULES
        },
        "examples": {
            "name": "examples",
            "type": "system_prompt",
            "description": "Examples of good and bad variable improvements.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_IMPROVEMENT_EXAMPLES
        },
        "output": {
            "name": "output",
            "type": "system_prompt",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_IMPROVEMENT_OUTPUT
        }
    }
}

REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT = {
    "name": "reflection_optimizer_improvement_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for self-improvement optimizer",
    "require_grad": False,
    "template": REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "Describes the task to be executed.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "current_variables": {
            "name": "current_variables",
            "type": "agent_message_prompt",
            "description": "Describes the current variables that need to be improved.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "reflection_analysis": {
            "name": "reflection_analysis",
            "type": "agent_message_prompt",
            "description": "Describes the reflection analysis for the current variables.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "memory_context": {
            "name": "memory_context",
            "type": "agent_message_prompt",
            "description": "Summaries and insights from previous optimization sessions.",
            "require_grad": False,
            "template": None,
            "variables": None
        }
    }
}

@PROMPT.register_module(force=True)
class ReflectionOptimizerImprovementSystemPrompt(Prompt):
    """System prompt template for self-improvement optimizer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='system_prompt', description="The type of the prompt")
    name: str = Field(default="reflection_optimizer_improvement", description="The name of the prompt")
    description: str = Field(default="System prompt for self-improvement optimizer", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=REFLECTION_OPTIMIZER_IMPROVEMENT_SYSTEM_PROMPT, description="System prompt information")

@PROMPT.register_module(force=True)
class ReflectionOptimizerImprovementAgentMessagePrompt(Prompt):
    """Agent message prompt template for self-improvement optimizer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='agent_message_prompt', description="The type of the prompt")
    name: str = Field(default="reflection_optimizer_improvement", description="The name of the prompt")
    description: str = Field(default="Agent message prompt for self-improvement optimizer", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT, description="Agent message prompt information")