from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict
from pydantic import Field, ConfigDict

# ---------------------Reflection Prompt---------------------
REFLECTION_OPTIMIZER_REFLECTION_AGENT_PROFILE = """
You are an expert at analyzing agent execution results and identifying areas for improvement.
"""

REFLECTION_OPTIMIZER_REFLECTION_INTRODUCTION = """
<intro>
You excel at:
- Analyzing agent execution results and identifying areas for improvement
- Reflecting on how the current prompt might have contributed to these issues
- Providing specific, actionable feedback on how to improve the prompt
- Being constructive and specific
- Providing concrete suggestions for improvement
</intro>
"""

REFLECTION_OPTIMIZER_REFLECTION_REASONING_RULES = """
<reasoning_rules>
Please analyze the execution result and provide:
- **What went wrong or could be improved?**
   - Identify specific issues in the agent's behavior or output
   - Note any errors, inefficiencies, or suboptimal outcomes

- **How did the prompt contribute to these issues?**
   - Identify parts of the prompt that may have led to the problems
   - Note any unclear, ambiguous, or missing instructions

- **Specific recommendations for improving the prompt**
   - Provide concrete suggestions for changes
   - Focus on making instructions clearer, more specific, and better structured
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
    "variables": [
        {
            "name": "agent_profile",
            "type": "system_prompt_module",
            "description": "Describes the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_REFLECTION_AGENT_PROFILE
        },
        {
            "name": "introduction",
            "type": "system_prompt_module",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_REFLECTION_INTRODUCTION
        },
        {
            "name": "reasoning_rules",
            "type": "system_prompt_module",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_REFLECTION_REASONING_RULES
        },
        {
            "name": "output",
            "type": "system_prompt_module",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_REFLECTION_OUTPUT
        }
    ]
}
REFLECTION_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT = {
    "name": "reflection_optimizer_reflection_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for self-reflection optimizer",
    "template": REFLECTION_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": [
        {
            "name": "task",
            "type": "agent_message_prompt_module",
            "description": "Describes the task to be executed.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "variable_to_improve",
            "type": "agent_message_prompt_module",
            "description": "Describes the variable to be improved.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "execution_result",
            "type": "agent_message_prompt_module",
            "description": "Describes the agent execution result.",
            "require_grad": False,
            "template": None,
            "variables": None
        }
    ]
}

@PROMPT.register_module(force=True)
class ReflectionOptimizerReflectionPrompt(Prompt):
    """Prompt template for self-reflection optimizer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="reflection_optimizer_reflection_prompt", description="The name of the prompt")
    description: str = Field(default="Prompt for self-reflection optimizer", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    @property
    def system_prompt(self) -> Dict[str, Any]:
        return REFLECTION_OPTIMIZER_REFLECTION_SYSTEM_PROMPT
    
    @property
    def agent_message_prompt(self) -> Dict[str, Any]:
        return REFLECTION_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT
# ---------------------Reflection Prompt---------------------

# ---------------------Improvement Prompt---------------------
REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_PROFILE = """
You are an expert at improving prompts based on feedback and analysis.
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_INTRODUCTION = """
<intro>
You excel at:
- Improving prompts based on feedback and analysis
- Keeping the core purpose and structure of the original prompt
- Addressing all identified issues
- Making instructions clearer, more specific, and better organized
- Removing unnecessary or confusing elements
- Adding missing guidance that would help the agent perform better
</intro>
"""

REFLECTION_OPTIMIZER_IMPROVEMENT_REASONING_RULES = """
<reasoning_rules>
Based on the analysis and feedback provided, please improve the prompt by:

1. **Keep the core purpose and structure**
   - Maintain the original prompt's fundamental purpose
   - Preserve the overall structure unless it's part of the problem

2. **Address all identified issues**
   - Systematically address each issue mentioned in the analysis
   - Ensure no problems are left unresolved

3. **Make instructions clearer and more specific**
   - Replace vague language with precise, actionable instructions
   - Add concrete examples where helpful
   - Clarify ambiguous requirements

4. **Better organization**
   - Improve the logical flow of instructions
   - Group related concepts together
   - Use clear headings and structure

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
Please provide ONLY the improved prompt text, without any additional commentary or explanation.
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
    "variables": [
        {
            "name": "agent_profile",
            "type": "system_prompt_module",
            "description": "Describes the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_PROFILE
        },
        {
            "name": "introduction",
            "type": "system_prompt_module",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_IMPROVEMENT_INTRODUCTION
        },
        {
            "name": "reasoning_rules",
            "type": "system_prompt_module",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_IMPROVEMENT_REASONING_RULES
        },
        {
            "name": "output",
            "type": "system_prompt_module",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": REFLECTION_OPTIMIZER_IMPROVEMENT_OUTPUT
        }
    ]
}

REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT = {
    "name": "reflection_optimizer_improvement_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for self-improvement optimizer",
    "template": REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": [
        {
            "name": "task",
            "type": "agent_message_prompt_module",
            "description": "Describes the task to be executed.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "current_variable",
            "type": "agent_message_prompt_module",
            "description": "Describes the current variable.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "reflection_analysis",  
            "type": "agent_message_prompt_module",
            "description": "Describes the reflection analysis.",
            "require_grad": False,
            "template": None,
            "variables": None
        }
    ]
}

@PROMPT.register_module(force=True)
class ReflectionOptimizerImprovementPrompt(Prompt):
    """Prompt template for self-improvement optimizer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="reflection_optimizer_improvement_prompt", description="The name of the prompt")
    description: str = Field(default="Prompt for self-improvement optimizer", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    @property
    def system_prompt(self) -> Dict[str, Any]:
        return REFLECTION_OPTIMIZER_IMPROVEMENT_SYSTEM_PROMPT
    
    @property
    def agent_message_prompt(self) -> Dict[str, Any]:
        return REFLECTION_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT