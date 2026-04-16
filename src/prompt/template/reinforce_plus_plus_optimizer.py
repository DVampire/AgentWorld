from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict, Literal
from pydantic import Field, ConfigDict

# ---------------------REINFORCE++ Reflection Prompt---------------------

REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_AGENT_PROFILE = """
You are an expert at analyzing agent execution results and RL metrics to identify which variables (prompts, tools, solutions, etc.) need improvement.
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_INTRODUCTION = """
<intro>
You excel at:
- Analyzing agent execution results combined with RL metrics (reward, advantage, objective, policy ratio)
- Reflecting on how different types of variables (prompt variables, tool code, solution) contributed to issues
- Providing specific, actionable feedback on how to improve each variable type
- Being constructive and specific
- Providing concrete suggestions for improving variables based on their type
</intro>
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_REASONING_RULES = """
<reasoning_rules>
**RL Metrics Interpretation:**
- **Reward (0.0-1.0)**: Measures task completion quality (1.0 = perfect, 0.0 = failed)
- **Advantage**: Reward minus KL penalty (positive = good performance, negative = needs improvement)
- **Policy Ratio**: Similarity between current and previous solution (1.0 = identical, <1.0 = different)
- **Objective**: Clipped policy ratio × advantage (REINFORCE++ optimization target)

Based on the current RL metrics and execution results, analyze:

IMPORTANT: You can ONLY analyze and suggest improvements for variables that are listed in <current_variables>. 
Do NOT suggest creating new variables or tools that don't exist. Work with what is provided.

Please analyze the execution result and the provided variables in <current_variables>, then provide:
- **What went wrong or could be improved**
   - Identify specific issues in the agent's behavior or output
   - Note any errors, inefficiencies, or suboptimal outcomes

- **Which variables (from current_variables) contributed to these issues?**
   - Analyze prompt variables (system_prompt, agent_message_prompt and their sub-variables): identify unclear instructions, missing context, or structural issues
   - Analyze tool variables (tool_code): identify bugs, missing functionality, or incorrect logic
   - Analyze solution variables: identify if the solution approach itself needs improvement
   - Determine which specific variable(s) from <current_variables> are most likely causing the problems

- **Specific recommendations for improving each problematic variable**
   - You MUST only recommend improvements for variables that exist in <current_variables>
   - For prompt variables: provide concrete suggestions for clearer instructions, better structure, or additional context
   - For tool variables: provide specific code fixes, feature additions, or logic corrections
   - For solution variables: suggest alternative approaches or improvements to the solution strategy, ensuring the format meets task requirements and can be correctly parsed
   - Focus on making each variable type more effective for the given task
</reasoning_rules>
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_OUTPUT = """
<output>
Please ONLY respond with the text of the analysis and recommendations, without any additional commentary or explanation.
</output>
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ introduction }}
{{ reasoning_rules }}
{{ output }}
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT_TEMPLATE = """
{{ task }}
{{ current_variables }}
{{ execution_result }}

**RL Metrics:**
- Reward: {{ reward }}
- Advantage: {{ advantage }}
- Policy Ratio: {{ policy_ratio }}
- Objective: {{ objective }}
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_SYSTEM_PROMPT = {
    "name": "reinforce_plus_plus_optimizer_reflection_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for REINFORCE++ reflection optimizer",
    "template": REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Describes the agent's core identity and RL expertise.",
            "require_grad": False,
            "template": None,
            "variables": REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_AGENT_PROFILE
        },
        "introduction": {
            "name": "introduction",
            "type": "system_prompt",
            "description": "Defines the agent's capabilities in RL-based optimization.",
            "require_grad": False,
            "template": None,
            "variables": REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_INTRODUCTION
        },
        "reasoning_rules": {
            "name": "reasoning_rules",
            "type": "system_prompt",
            "description": "Defines how to interpret RL metrics and make optimization decisions.",
            "require_grad": False,
            "template": None,
            "variables": REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_REASONING_RULES
        },
        "output": {
            "name": "output",
            "type": "system_prompt",
            "description": "Defines the output format for RL-based analysis.",
            "require_grad": False,
            "template": None,
            "variables": REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_OUTPUT
        }
    }
}

REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT = {
    "name": "reinforce_plus_plus_optimizer_reflection_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for REINFORCE++ reflection optimizer",
    "require_grad": False,
    "template": REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT_TEMPLATE,
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
        "reward": {
            "name": "reward",
            "type": "agent_message_prompt",
            "description": "Current reward score from RL evaluation.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "advantage": {
            "name": "advantage",
            "type": "agent_message_prompt",
            "description": "Calculated advantage (reward - KL penalty).",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "policy_ratio": {
            "name": "policy_ratio",
            "type": "agent_message_prompt",
            "description": "Policy ratio between current and old solution.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "objective": {
            "name": "objective",
            "type": "agent_message_prompt",
            "description": "REINFORCE++ objective value.",
            "require_grad": False,
            "template": None,
            "variables": None
        }
    }
}

@PROMPT.register_module(force=True)
class ReinforcePlusPlusOptimizerReflectionSystemPrompt(Prompt):
    """System prompt template for REINFORCE++ reflection optimizer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default='system_prompt', description="The type of the prompt")
    name: str = Field(default="reinforce_plus_plus_optimizer_reflection", description="The name of the prompt")
    description: str = Field(default="System prompt for REINFORCE++ reflection optimizer", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")

    prompt_config: Dict[str, Any] = Field(default=REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_SYSTEM_PROMPT, description="System prompt information")

@PROMPT.register_module(force=True)
class ReinforcePlusPlusOptimizerReflectionAgentMessagePrompt(Prompt):
    """Agent message prompt template for REINFORCE++ reflection optimizer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default='agent_message_prompt', description="The type of the prompt")
    name: str = Field(default="reinforce_plus_plus_optimizer_reflection", description="The name of the prompt")
    description: str = Field(default="Agent message prompt for REINFORCE++ reflection optimizer", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")

    prompt_config: Dict[str, Any] = Field(default=REINFORCE_PLUS_PLUS_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT, description="Agent message prompt information")

# ---------------------REINFORCE++ Improvement Prompt---------------------

REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_AGENT_PROFILE = """
You are an expert at improving variables (prompts, tools, solutions) based on feedback and analysis.
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_INTRODUCTION = """
<intro>
You excel at:
- Improving ONLY the variables provided in <current_variables> based on feedback and analysis
- Keeping the core purpose and structure of the original variable
- Addressing all identified issues specific to each variable type
- Making improvements appropriate for the variable type (clearer instructions for prompts, bug fixes for tools, better strategies for solutions)
- Removing unnecessary or problematic elements
- Adding missing elements that would help the agent perform better

IMPORTANT: You can ONLY improve variables that are listed in <current_variables>. 
Do NOT suggest or create new variables that don't exist in the provided list.
</intro>
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_REASONING_RULES = """
<reasoning_rules>
Based on the analysis and feedback provided, please improve the variable by:

1. **Keep the core purpose and structure**
   - Maintain the original variable's fundamental purpose
   - Preserve the overall structure unless it's part of the problem

2. **Address all identified issues**
   - Systematically address each issue mentioned in the analysis
   - Ensure no problems are left unresolved

3. **Make improvements appropriate for the variable type**
   - For prompt variables: Make instructions clearer and more specific, replace vague language with precise instructions, add concrete examples, clarify ambiguous requirements. When improving prompts intended for problem-solving, consider phrasing that activates the model's reasoning.
   - For tool variables: Fix bugs, correct logic errors, add missing functionality, improve error handling
   - For solution variables: Suggest better approaches, improve strategy, refine the solution method

4. **Better organization**
   - For prompts: Improve the logical flow of instructions, group related concepts together, use clear headings and structure
   - For tools: Organize code better, improve readability, add proper documentation
   - For solutions: Structure the approach more clearly, break down complex steps, ensuring the format meets task requirements and can be correctly parsed

5. **Remove unnecessary elements**
   - Eliminate redundant or confusing parts
   - Streamline the prompt for clarity

6. **Add missing guidance**
   - Include any critical instructions that were missing
   - Add context that would help the agent perform better
</reasoning_rules>
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_OUTPUT = """
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

REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ introduction }}
{{ reasoning_rules }}
{{ output }}
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT_TEMPLATE = """
{{ task }}
{{ current_variables }}
{{ reflection_analysis }}
"""

REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_SYSTEM_PROMPT = {
    "name": "reinforce_plus_plus_optimizer_improvement_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for REINFORCE++ improvement optimizer",
    "template": REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Describes the agent's RL-guided optimization expertise.",
            "require_grad": False,
            "template": None,
            "variables": REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_AGENT_PROFILE
        },
        "introduction": {
            "name": "introduction",
            "type": "system_prompt",
            "description": "Defines RL-guided optimization capabilities.",
            "require_grad": False,
            "template": None,
            "variables": REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_INTRODUCTION
        },
        "reasoning_rules": {
            "name": "reasoning_rules",
            "type": "system_prompt",
            "description": "Defines how to use RL metrics for optimization decisions.",
            "require_grad": False,
            "template": None,
            "variables": REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_REASONING_RULES
        },
        "output": {
            "name": "output",
            "type": "system_prompt",
            "description": "Defines the output format for RL-guided improvements.",
            "require_grad": False,
            "template": None,
            "variables": REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_OUTPUT
        }
    }
}

REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT = {
    "name": "reinforce_plus_plus_optimizer_improvement_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for REINFORCE++ improvement optimizer",
    "require_grad": False,
    "template": REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT_TEMPLATE,
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
            "description": "Describes the RL-guided reflection analysis for the current variables.",
            "require_grad": False,
            "template": None,
            "variables": None
        }
    }
}

@PROMPT.register_module(force=True)
class ReinforcePlusPlusOptimizerImprovementSystemPrompt(Prompt):
    """System prompt template for REINFORCE++ improvement optimizer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default='system_prompt', description="The type of the prompt")
    name: str = Field(default="reinforce_plus_plus_optimizer_improvement", description="The name of the prompt")
    description: str = Field(default="System prompt for REINFORCE++ improvement optimizer", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")

    prompt_config: Dict[str, Any] = Field(default=REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_SYSTEM_PROMPT, description="System prompt information")

@PROMPT.register_module(force=True)
class ReinforcePlusPlusOptimizerImprovementAgentMessagePrompt(Prompt):
    """Agent message prompt template for REINFORCE++ improvement optimizer."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    type: str = Field(default='agent_message_prompt', description="The type of the prompt")
    name: str = Field(default="reinforce_plus_plus_optimizer_improvement", description="The name of the prompt")
    description: str = Field(default="Agent message prompt for REINFORCE++ improvement optimizer", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")

    prompt_config: Dict[str, Any] = Field(default=REINFORCE_PLUS_PLUS_OPTIMIZER_IMPROVEMENT_AGENT_MESSAGE_PROMPT, description="Agent message prompt information")
