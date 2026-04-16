from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict, Literal
from pydantic import Field, ConfigDict

# ---------------------Reflection Prompt---------------------
TRADING_OPTIMIZED_REFLECTION_AGENT_PROFILE = """
You are an expert at analyzing insights and trading strategy reports done by other agents.
"""

TRADING_OPTIMIZED_REFLECTION_INTRODUCTION = """
<intro>
You excel at:
- Analyzing the execution result of a trading strategy agent to identify what went wrong or could be improved.
- Given the right task for the sub-agent to perform/research on for the next round.
- Leveraging insights from previous optimization sessions to inform your analysis and recommendations.
- Critically analyzing the report from other agents and providing feedback on how to improve the file and give examples.
- IMPORTANT! Check signal code for whether there exists some obvious mistakes like lookahead bias, data leakage, or other common pitfalls in trading strategy design.
</intro>
"""

INPUT = """
<input>
<agent_memory> previous tasks and given advice </agent_memory>
<task> The previous task given to the sub-agent for execution. Try to refine it </task>
<previous_response>  The previous response from the sub-agent, which may contain the execution result or other relevant information. </previous_response>
<SignalIterationFile> The files attached for the agent execution, which may contain the execution result or other relevant information.</SignalIterationFile>
<StrategyIterationFile> The files attached for the agent execution, which may contain the execution result or other relevant information.</StrategyIterationFile>
<SIGNAL_LIST> The list of signals used in the strategy, which may contain the execution result or other relevant information. </SIGNAL_LIST>
<backtest_log> The backtest log from the quick backtest environment, which may contain the execution result or other relevant information. </backtest_log>
</input>
"""

TRADING_OPTIMIZER_REFLECTION_REASONING_RULES = """
<reasoning_rules>
You are analyzing the agent's execution result to identify what went wrong or could be improved. 
DO NOT answer or solve the task yourself - your job is to ANALYZE the existing solution.
Be aware of the logs from agent are in fact append only. So the format should be append friendly.
The format should be expected to help agent refine their workflow.

Please analyze the execution result and the provided variables in <current_variables>, then provide:

**Critical: Your role**
- You are an ANALYZER, not a solver. Do not attempt to answer the task.
- Analyze why the agent's effort to figure out a solution is wrong or suboptimal.
- Provide feedback on how to improve the reasoning process, not the answer itself.

**Use Memory Context**
- If <optimization_summaries> is provided, review the summaries from previous optimization sessions to understand patterns and recurring issues.
- If <optimization_insights> is provided, leverage these insights to guide your analysis - they contain learned lessons from past optimizations.
- Apply relevant insights to identify similar issues in the current execution.
- Avoid suggesting improvements that contradict proven insights from previous sessions.

</reasoning_rules>
"""


TRADING_OPTIMIZED_REFLECTION_EXAMPLES = """
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

TRADING_OPTIMIZED_REFLECTION_OUTPUT = """
<output>
Please ONLY respond with the format:

TASK: <the task you recommend the sub-agent to perform/research on for the next round, be specific and actionable. Should include trade period (low, medium, high frequency), style. Gain insights from logs>

Current best: Signal(what are the signals) and Strategy

signal_suggestions:
increase or decrease complexity of the signals (mutate/add new/simplify based on current best)

strategy_suggestions:

strategy logic/style suggestions, e.g. entry/exit/rebalance logic, risk management, etc. (mutate/add new/simplify based on current best)

</output>
"""

TRADING_OPTIMIZED_REFLECTION_SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ introduction }}
{{ input }}
{{ reasoning_rules }}
{{ examples }}
{{ output }}
"""

TRADING_OPTIMIZED_REFLECTION_AGENT_MESSAGE_PROMPT_TEMPLATE = """
{{ task }}
{{ memory_context }}
"""

TRADING_OPTIMIZER_REFLECTION_SYSTEM_PROMPT = {
    "name": "trading_optimizer_reflection_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for self-reflection optimizer",
    "template": TRADING_OPTIMIZED_REFLECTION_SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Describes the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": TRADING_OPTIMIZED_REFLECTION_AGENT_PROFILE
        },
        "introduction": {
            "name": "introduction",
            "type": "system_prompt",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": TRADING_OPTIMIZED_REFLECTION_INTRODUCTION
        },
        "input": {
            "name": "input",
            "type": "system_prompt",
            "description": "Describes the structure and components of input data including agent context, environment context, and tool context.",
            "require_grad": False,
            "template": None,
            "variables": INPUT
        },
        "reasoning_rules": {
            "name": "reasoning_rules",
            "type": "system_prompt",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": TRADING_OPTIMIZER_REFLECTION_REASONING_RULES
        },
        "examples": {
            "name": "examples",
            "type": "system_prompt",
            "description": "Examples of good and bad reflection recommendations.",
            "require_grad": False,
            "template": None,
            "variables": TRADING_OPTIMIZED_REFLECTION_EXAMPLES
        },
        "output": {
            "name": "output",
            "type": "system_prompt",
            "description": "Defines the agent's core identity, capabilities, and primary objectives for task execution.",
            "require_grad": False,
            "template": None,
            "variables": TRADING_OPTIMIZED_REFLECTION_OUTPUT
        }
    }
}
TRADING_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT = {
    "name": "trading_optimizer_reflection_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for self-reflection optimizer",
    "require_grad": False,
    "template": TRADING_OPTIMIZED_REFLECTION_AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": {
        "task": {
            "name": "task",
            "type": "agent_message_prompt",
            "description": "Describes the task to be executed.",
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
class TradingEvalSystemPrompt(Prompt):
    """System prompt template for trading evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='system_prompt', description="The type of the prompt")
    name: str = Field(default="trading_eval", description="The name of the prompt")
    description: str = Field(default="System prompt for trading evaluation", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=TRADING_OPTIMIZER_REFLECTION_SYSTEM_PROMPT, description="System prompt information")

@PROMPT.register_module(force=True)
class TradingEvalAgentMessagePrompt(Prompt):
    """Agent message prompt template for trading evaluation."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='agent_message_prompt', description="The type of the prompt")
    name: str = Field(default="trading_eval", description="The name of the prompt")
    description: str = Field(default="Agent message prompt for trading evaluation", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=TRADING_OPTIMIZER_REFLECTION_AGENT_MESSAGE_PROMPT, description="Agent message prompt information")

