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
Please analyze the execution result and all available variables, then provide:
- **What went wrong or could be improved**
   - Identify specific issues in the agent's behavior or output
   - Note any errors, inefficiencies, or suboptimal outcomes

- **Which variables contributed to these issues?**
   - Analyze prompt variables (system_prompt, agent_message_prompt and their sub-variables): identify unclear instructions, missing context, or structural issues
   - Analyze tool variables (tool_code): identify bugs, missing functionality, or incorrect logic
   - Analyze solution variables: identify if the solution approach itself needs improvement
   - Determine which specific variable(s) are most likely causing the problems

- **Specific recommendations for improving each problematic variable**
   - For prompt variables: provide concrete suggestions for clearer instructions, better structure, or additional context
   - For tool variables: provide specific code fixes, feature additions, or logic corrections
   - For solution variables: suggest alternative approaches or improvements to the solution strategy
   - Focus on making each variable type more effective for the given task
</reasoning_rules>
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
{{ output }}
"""

REFLECTION_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT_TEMPLATE = f"""
{{ task }}
{{ variable_to_improve }}
{{ execution_result }}
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
        "variable_to_improve": {
            "name": "variable_to_improve",
            "type": "agent_message_prompt",
            "description": "Describes the variable to be improved.",
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
- Improving different types of variables (prompt variables, tool code, solutions) based on feedback and analysis
- Keeping the core purpose and structure of the original variable
- Addressing all identified issues specific to each variable type
- Making improvements appropriate for the variable type (clearer instructions for prompts, bug fixes for tools, better strategies for solutions)
- Removing unnecessary or problematic elements
- Adding missing elements that would help the agent perform better
</intro>
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_REASONING_RULES = """
<reasoning_rules>
Based on the analysis and feedback provided, please improve the variable by:

1. **Keep the core purpose and structure**
   - Maintain the original variable's fundamental purpose
   - Preserve the overall structure unless it's part of the problem

2. **Address all identified issues**
   - Systematically address each issue mentioned in the analysis
   - Ensure no problems are left unresolved

3. **Make improvements appropriate for the variable type**
   - For prompt variables: Make instructions clearer and more specific, replace vague language with precise instructions, add concrete examples, clarify ambiguous requirements
   - For tool variables: Fix bugs, correct logic errors, add missing functionality, improve error handling
   - For solution variables: Suggest better approaches, improve strategy, refine the solution method

4. **Better organization**
   - For prompts: Improve the logical flow of instructions, group related concepts together, use clear headings and structure
   - For tools: Organize code better, improve readability, add proper documentation
   - For solutions: Structure the approach more clearly, break down complex steps

5. **Remove unnecessary elements**
   - Eliminate redundant or confusing parts
   - Streamline the prompt for clarity

6. **Add missing guidance**
   - Include any critical instructions that were missing
   - Add context that would help the agent perform better
</reasoning_rules>
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_OUTPUT = """
<output>
Please provide ONLY the improved variable content:
- For prompt variables: Provide the improved prompt text
- For tool variables: Provide the improved tool code
- For solution variables: Provide the improved solution approach
Do not include any additional commentary or explanation.
</output>
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ introduction }}
{{ reasoning_rules }}
{{ output }}
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT_TEMPLATE = f"""
{{ task }}
{{ current_variable }}
{{ reflection_analysis }}
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
        "current_variable": {
            "name": "current_variable",
            "type": "agent_message_prompt",
            "description": "Describes the current variable.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "reflection_analysis": {
            "name": "reflection_analysis",
            "type": "agent_message_prompt",
            "description": "Describes the reflection analysis.",
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