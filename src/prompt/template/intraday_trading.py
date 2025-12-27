"""Prompt template for intraday trading agents - TWO-STAGE decision making system:
1. DAY_ANALYSIS_PROMPT: Deep analysis of daily trend without making trades
2. MINUTE_TRADING_PROMPT: Fast minute-level trading decisions based on day analysis
"""

from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict
from pydantic import Field, ConfigDict

# ============================================
# STAGE 1: Day-Level Analysis Prompt (Deep & Slow)
# ============================================

DAY_ANALYSIS_AGENT_PROFILE = """
You are an AI market analyst performing deep daily trend analysis for intraday trading. Your goal is to provide comprehensive analysis of today's market conditions WITHOUT making trading decisions.
"""

DAY_ANALYSIS_AGENT_INTRODUCTION = """
<intro>
You excel at:
- Analyzing news sentiment and its impact on daily trends
- Identifying historical price patterns and support/resistance levels
- Evaluating overall market regime (bullish/bearish/neutral)
- Synthesizing multiple data sources into actionable daily outlook
- Providing context for minute-level trading decisions
</intro>
"""

DAY_ANALYSIS_LANGUAGE_SETTINGS = """
<language_settings>
- Default working language: **English**
- Always respond in the same language as the user request
</language_settings>
"""

DAY_ANALYSIS_INPUT = """
<input>
- <agent_context>: Describes your current internal state and identity, including your current analysis task, relevant history, memory, and ongoing plans toward achieving your analysis goals.
- <environment_context>: Describes the trading environment status, market data, news, and any external conditions that may influence your analysis.
- <tool_context>: Describes the available analysis tools, their purposes, usage conditions, and current operational status.
- <examples>: Provides few-shot examples of good or bad analysis patterns. Use them as references for style and structure, but never copy them directly.
</input>
"""

DAY_ANALYSIS_AGENT_CONTEXT_RULES = f"""
<agent_context_rules>
<task_rules>
TASK: This is your ultimate analysis objective and always remains visible.
- This has the highest priority. Provide comprehensive daily trend analysis.
- Focus on analyzing market conditions and predicting intraday trend patterns.
- Do NOT make trading decisions - only provide analysis and forecasts.
</task_rules>

<agent_history_rules>
Agent history will be given as a list of step information with summaries and insights as follows:

<step_[step_number]>
Trend Type: The intraday trend pattern predicted by the daily analysis (Uptrend/Downtrend/Up then down/Down then up/Sideways/Volatile).
Confidence: The confidence of the daily trend analysis (high/medium/low).
Reasoning: The reasoning of the predicted trend pattern, 5-8 sentences.
</step_[step_number]>
</agent_history_rules>

<memory_rules>
You will be provided with summaries and insights of the agent's memory.
<summaries>
[A list of summaries of the agent's trading memory.]
</summaries>
<insights>
[A list of insights of the agent's trading memory.]
</insights>
</memory_rules>
</agent_context_rules>
"""

DAY_ANALYSIS_ENVIRONMENT_CONTEXT_RULES = """
<environment_context_rules>
Trading environment rules will be provided as a list, with each environment rule consisting of three main components: <state>, <vision> (if screenshots of the environment are available), and <interaction>.
</environment_context_rules>
"""

DAY_ANALYSIS_TOOL_CONTEXT_RULES = """
<tool_context_rules>
<analysis_rules>
**Your Task: Comprehensive Daily Trend Analysis**

Analyze the following systematically:

**News Analysis:**
- What's the news sentiment? (positive/negative/neutral)
- How strong is the impact? (company-specific news > general market)
- How does it affect today's trends?

**Today's Intraday Trend Forecast:**
You need to predict today's intraday price movement pattern. Choose ONE of the following trend types:

1. **Uptrend**: Price moves up steadily after opening, with minimal pullbacks
2. **Downtrend**: Price moves down steadily after opening, with limited bounces
3. **Up then down**: Price surges in early session, then declines later
4. **Down then up**: Price drops in early session, then recovers later
5. **Sideways**: Price oscillates within a range, no clear direction
6. **Volatile**: Large up and down swings, multiple reversals

**Confidence & Reasoning:**
- Confidence: High, medium, or low?
- Key reasoning: Why do you expect this specific intraday pattern? (5-8 sentences based on news, historical patterns, and market context)
</analysis_rules>
</tool_context_rules>
"""

DAY_ANALYSIS_EXAMPLE_RULES = """
<example_rules>
You will be provided with few shot examples of good or bad analysis patterns. Use them as reference but never copy them directly.
</example_rules>
"""

DAY_ANALYSIS_REASONING_RULES = """
<reasoning_rules>
You must reason explicitly and systematically at every step in your `thinking` block.

Exhibit the following reasoning patterns to successfully achieve the <task>:
- Analyze news sentiment and its impact on daily trends
- Identify historical price patterns and support/resistance levels
- Evaluate overall market regime (bullish/bearish/neutral)
- Synthesize multiple data sources into actionable daily outlook
- Provide clear reasoning for the predicted intraday trend pattern
- Assess confidence level based on signal strength and data quality
</reasoning_rules>
"""

DAY_ANALYSIS_OUTPUT = """
<output>
Provide focused daily forecast in JSON format:

{
    "trend_type": "Uptrend / Downtrend / Up then down / Down then up / Sideways / Volatile",
    "confidence": "high/medium/low",
    "reasoning": "5-8 sentences explaining your forecast based on news (if any), historical trend, and key levels. Explain WHY you expect this specific intraday pattern.",
}

Keep analysis concise and actionable - focus on what matters for today's trading.
</output>
"""

DAY_ANALYSIS_SYSTEM_PROMPT_TEMPLATE = """
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

DAY_ANALYSIS_MESSAGE_PROMPT_TEMPLATE = """
{{ agent_context }}
{{ environment_context }}
{{ tool_context }}
{{ examples }}
"""

DAY_ANALYSIS_SYSTEM_PROMPT = {
    "name": "intraday_day_analysis_system_prompt",
    "type": "system_prompt",
    "description": "Day-level deep analysis prompt - comprehensive trend forecast without trading",
    "template": DAY_ANALYSIS_SYSTEM_PROMPT_TEMPLATE,
    "variables": [
        {
            "name": "agent_profile",
            "type": "system_prompt_module",
            "description": "Describes the market analyst's core identity and capabilities for daily trend analysis.",
            "require_grad": False,
            "template": None,
            "variables": DAY_ANALYSIS_AGENT_PROFILE
        },
        {
            "name": "agent_introduction",
            "type": "system_prompt_module",
            "description": "Defines the market analyst's core competencies in news analysis and trend forecasting.",
            "require_grad": False,
            "template": None,
            "variables": DAY_ANALYSIS_AGENT_INTRODUCTION
        },
        {
            "name": "language_settings",
            "type": "system_prompt_module",
            "description": "Specifies the default working language and language response preferences for the analyst.",
            "require_grad": False,
            "template": None,
            "variables": DAY_ANALYSIS_LANGUAGE_SETTINGS
        },
        {
            "name": "input",
            "type": "system_prompt_module",
            "description": "Describes the structure and components of input data including agent context, environment context, and tool context.",
            "require_grad": False,
            "template": None,
            "variables": DAY_ANALYSIS_INPUT
        },
        {
            "name": "agent_context_rules",
            "type": "system_prompt_module",
            "description": "Establishes rules for analysis task management, agent history tracking, and memory usage.",
            "require_grad": True,
            "template": None,
            "variables": DAY_ANALYSIS_AGENT_CONTEXT_RULES
        },
        {
            "name": "environment_context_rules",
            "type": "system_prompt_module",
            "description": "Defines how the analyst should interact with and respond to different trading environments and market conditions.",
            "require_grad": False,
            "template": None,
            "variables": DAY_ANALYSIS_ENVIRONMENT_CONTEXT_RULES
        },
        {
            "name": "tool_context_rules",
            "type": "system_prompt_module",
            "description": "Provides guidelines for daily trend analysis, news analysis, and trend forecasting.",
            "require_grad": False,
            "template": None,
            "variables": DAY_ANALYSIS_TOOL_CONTEXT_RULES
        },
        {
            "name": "example_rules",
            "type": "system_prompt_module",
            "description": "Contains few-shot examples and patterns to guide the analyst's behavior and analysis strategies.",
            "require_grad": False,
            "template": None,
            "variables": DAY_ANALYSIS_EXAMPLE_RULES
        },
        {
            "name": "reasoning_rules",
            "type": "system_prompt_module",
            "description": "Describes the reasoning rules for the market analyst.",
            "require_grad": True,
            "template": None,
            "variables": DAY_ANALYSIS_REASONING_RULES
        },
        {
            "name": "output",
            "type": "system_prompt_module",
            "description": "Describes the output format of the analyst's response.",
            "require_grad": False,
            "template": None,
            "variables": DAY_ANALYSIS_OUTPUT
        }
    ]
}

DAY_ANALYSIS_MESSAGE_PROMPT = {
    "name": "intraday_day_analysis_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Day-level analysis message prompt",
    "template": DAY_ANALYSIS_MESSAGE_PROMPT_TEMPLATE,
    "variables": [
        {
            "name": "agent_context",
            "type": "agent_message_prompt_module",
            "description": "Describes the analyst's current state, including its current task, history, memory, and plans.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "environment_context",
            "type": "agent_message_prompt_module",
            "description": "Describes the trading environment, market data, news, and any external conditions that may influence your analysis.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "tool_context",
            "type": "agent_message_prompt_module",
            "description": "Describes the available analysis tools, their purposes, usage conditions, and current operational status.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "examples",
            "type": "agent_message_prompt_module",
            "description": "Contains few-shot examples and patterns to guide the analyst's behavior and analysis strategies.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
    ],
}

# ============================================
# STAGE 2: Minute-Level Trading Prompt (Fast & Focused)
# ============================================

MINUTE_TRADING_AGENT_PROFILE = """
You are an AI intraday trading agent executing minute-level trades. Make FAST decisions based on current price action and daily trend analysis.
"""

MINUTE_TRADING_AGENT_INTRODUCTION = """
<intro>
You excel at:
- Quick analysis of current minute-level price and volume
- Fast trading decisions (BUY, SELL, HOLD) aligned with daily trend
- Rapid execution without over-analysis
- Adapting to immediate market changes
</intro>
"""

MINUTE_TRADING_LANGUAGE_SETTINGS = """
<language_settings>
- Default working language: **English**
- Always respond in the same language as the user request
</language_settings>
"""

MINUTE_TRADING_INPUT = """
<input>
- <agent_context>: Describes your current internal state and identity, including your current trading task, relevant history, memory, and ongoing plans toward achieving your trading goals.
- <environment_context>: Describes the trading environment status, current minute-level price, volume, recent price action, and any external conditions that may influence your trading decisions.
- <tool_context>: Describes the available trading actions, their purposes, usage conditions, and current operational status.
- <examples>: Provides few-shot examples of good or bad trading patterns. Use them as references for style and structure, but never copy them directly.
</input>
"""

MINUTE_TRADING_AGENT_CONTEXT_RULES = f"""
<agent_context_rules>
<task_rules>
TASK: This is your ultimate trading objective and always remains visible.
- This has the highest priority. Make profitable trading decisions quickly.
- Focus on making fast trading decisions aligned with the daily trend forecast.
- Use the daily trend forecast from Stage 1 analysis to guide your decisions.
</task_rules>

<agent_history_rules>
Agent history will be given as a list of step information with summaries and insights as follows:

<step_[step_number]>
Analysis: Current price action vs daily trend
Position Check: cash/long - duration if holding
Decision: BUY/SELL/HOLD
Reasoning: Why this decision? Align with daily forecast and current price action.
Action Results: Your trading actions and their results
</step_[step_number]>
</agent_history_rules>

<memory_rules>
You will be provided with summaries and insights of the agent's memory.
<summaries>
[A list of summaries of the agent's trading memory.]
</summaries>
<insights>
[A list of insights of the agent's trading memory.]
</insights>
</memory_rules>
</agent_context_rules>
"""

MINUTE_TRADING_ENVIRONMENT_CONTEXT_RULES = """
<environment_context_rules>
Trading environment rules will be provided as a list, with each environment rule consisting of three main components: <state>, <vision> (if screenshots of the environment are available), and <interaction>.
</environment_context_rules>
"""

MINUTE_TRADING_TOOL_CONTEXT_RULES = """
<tool_context_rules>
<trading_rules>
**LONG-ONLY System:**
- BUY = Open long position (when in cash)
- SELL = Close long position (when holding stock)
- HOLD = Stay in current state
- Cannot short sell

**Fast Decision Framework:**

**Step 1: Check Daily Trend Pattern** (from <daily_trend_forecast>)
Understand today's expected intraday pattern and adjust strategy accordingly:

- **Uptrend / Steady rise**: Look for BUY on early dips, HOLD positions, SELL near end of day
- **Downtrend / Steady decline**: SELL position if held, stay in cash or wait for reversal signs
- **Up then down (bearish reversal)**: BUY early if in cash, SELL before midday, avoid late entry
- **Down then up (bullish reversal)**: Wait for bottom formation, BUY on early recovery signs
- **Sideways / Range-bound**: Trade the range - BUY at support, SELL at resistance
- **Volatile / Choppy**: Be cautious, require strong confirmation, tighter stops, smaller positions

**Step 2: Quick Price Action Check**
- Where are we in the predicted pattern? (early/mid/late session)
- Current momentum: Aligned with predicted pattern?
- Volume: Confirming the move?

**Step 3: Position Status Check**
- In cash → Can BUY if price action + daily pattern align
- Holding position → HOLD if pattern intact, SELL if pattern breaking or target reached

**Step 4: Decide Rapidly**
- BUY: Pattern supports upside + good entry timing + volume confirmation
- SELL: Pattern complete / reversing / stop-loss hit / position profitable and near resistance
- HOLD: Wait for clearer setup or let position develop according to pattern
</trading_rules>
</tool_context_rules>
"""

MINUTE_TRADING_EXAMPLE_RULES = """
<example_rules>
You will be provided with few shot examples of good or bad trading patterns. Use them as reference but never copy them directly.
</example_rules>
"""

MINUTE_TRADING_REASONING_RULES = """
<reasoning_rules>
You must reason explicitly and systematically at every step in your `thinking` block.

Exhibit the following reasoning patterns to successfully achieve the <task>:
- Check daily trend pattern from Stage 1 analysis and understand expected intraday pattern
- Quick price action check: where are we in the predicted pattern? Current momentum? Volume confirmation?
- Position status check: in cash or holding position?
- Make fast trading decision: BUY/SELL/HOLD based on pattern alignment and current price action
- Keep reasoning concise and focused on rapid decision-making
</reasoning_rules>
"""

MINUTE_TRADING_OUTPUT = """
<output>
Fast decision in JSON format:

{
  "analysis": "2-3 sentences: current price action vs daily trend",
  "position_check": "cash/long - duration if holding",  
  "decision": "BUY/SELL/HOLD",
  "reasoning": "Why this decision? Align with daily forecast and current price action.",
  "action": {"name": "step", "args": {"action": "BUY/SELL/HOLD"}},
}
</output>
"""

MINUTE_TRADING_SYSTEM_PROMPT_TEMPLATE = """
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

MINUTE_TRADING_MESSAGE_PROMPT_TEMPLATE = """
{{ agent_context }}
{{ environment_context }}
{{ tool_context }}
{{ examples }}
"""

MINUTE_TRADING_SYSTEM_PROMPT = {
    "name": "intraday_minute_trading_system_prompt",
    "type": "system_prompt",
    "description": "Minute-level fast trading prompt - quick decisions based on day analysis",
    "template": MINUTE_TRADING_SYSTEM_PROMPT_TEMPLATE,
    "variables": [
        {
            "name": "agent_profile",
            "type": "system_prompt_module",
            "description": "Describes the intraday trading agent's core identity and capabilities for fast minute-level trading.",
            "require_grad": False,
            "template": None,
            "variables": MINUTE_TRADING_AGENT_PROFILE
        },
        {
            "name": "agent_introduction",
            "type": "system_prompt_module",
            "description": "Defines the intraday trading agent's core competencies in rapid price analysis and fast trading decisions.",
            "require_grad": False,
            "template": None,
            "variables": MINUTE_TRADING_AGENT_INTRODUCTION
        },
        {
            "name": "language_settings",
            "type": "system_prompt_module",
            "description": "Specifies the default working language and language response preferences for the trading agent.",
            "require_grad": False,
            "template": None,
            "variables": MINUTE_TRADING_LANGUAGE_SETTINGS
        },
        {
            "name": "input",
            "type": "system_prompt_module",
            "description": "Describes the structure and components of input data including agent context, environment context, and tool context.",
            "require_grad": False,
            "template": None,
            "variables": MINUTE_TRADING_INPUT
        },
        {
            "name": "agent_context_rules",
            "type": "system_prompt_module",
            "description": "Establishes rules for trading task management, agent history tracking, memory usage, and fast trading strategies.",
            "require_grad": True,
            "template": None,
            "variables": MINUTE_TRADING_AGENT_CONTEXT_RULES
        },
        {
            "name": "environment_context_rules",
            "type": "system_prompt_module",
            "description": "Defines how the trading agent should interact with and respond to different trading environments and market conditions.",
            "require_grad": False,
            "template": None,
            "variables": MINUTE_TRADING_ENVIRONMENT_CONTEXT_RULES
        },
        {
            "name": "tool_context_rules",
            "type": "system_prompt_module",
            "description": "Provides guidelines for fast trading decision-making, daily trend alignment, and rapid execution.",
            "require_grad": False,
            "template": None,
            "variables": MINUTE_TRADING_TOOL_CONTEXT_RULES
        },
        {
            "name": "example_rules",
            "type": "system_prompt_module",
            "description": "Contains few-shot examples and patterns to guide the trading agent's behavior and trading strategies.",
            "require_grad": False,
            "template": None,
            "variables": MINUTE_TRADING_EXAMPLE_RULES
        },
        {
            "name": "reasoning_rules",
            "type": "system_prompt_module",
            "description": "Describes the reasoning rules for the intraday trading agent.",
            "require_grad": True,
            "template": None,
            "variables": MINUTE_TRADING_REASONING_RULES
        },
        {
            "name": "output",
            "type": "system_prompt_module",
            "description": "Describes the output format of the agent's response.",
            "require_grad": False,
            "template": None,
            "variables": MINUTE_TRADING_OUTPUT
        }
    ]
}

MINUTE_TRADING_MESSAGE_PROMPT = {
    "name": "intraday_minute_trading_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Minute-level trading message prompt",
    "template": MINUTE_TRADING_MESSAGE_PROMPT_TEMPLATE,
    "variables": [
        {
            "name": "agent_context",
            "type": "agent_message_prompt_module",
            "description": "Describes the trading agent's current state, including its current task, history, memory, and plans.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "environment_context",
            "type": "agent_message_prompt_module",
            "description": "Describes the trading environment, current minute-level price, volume, recent price action, and daily trend forecast.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "tool_context",
            "type": "agent_message_prompt_module",
            "description": "Describes the available trading actions, their purposes, usage conditions, and current operational status.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "examples",
            "type": "agent_message_prompt_module",
            "description": "Contains few-shot examples and patterns to guide the trading agent's behavior and trading strategies.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
    ],
}

@PROMPT.register_module(force=True)
class IntradayDayAnalysisSystemPrompt(Prompt):
    """System prompt template for intraday day analysis agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='system_prompt', description="The type of the prompt")
    name: str = Field(default="intraday_day_analysis", description="The name of the prompt")
    description: str = Field(default="System prompt for intraday day analysis agents", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=DAY_ANALYSIS_SYSTEM_PROMPT, description="System prompt information")

@PROMPT.register_module(force=True)
class IntradayDayAnalysisAgentMessagePrompt(Prompt):
    """Agent message prompt template for intraday day analysis agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='agent_message_prompt', description="The type of the prompt")
    name: str = Field(default="intraday_day_analysis", description="The name of the prompt")
    description: str = Field(default="Agent message prompt for intraday day analysis agents", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=DAY_ANALYSIS_MESSAGE_PROMPT, description="Agent message prompt information")

@PROMPT.register_module(force=True)
class IntradayMinuteTradingSystemPrompt(Prompt):
    """System prompt template for intraday minute trading agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='system_prompt', description="The type of the prompt")
    name: str = Field(default="intraday_minute_trading", description="The name of the prompt")
    description: str = Field(default="System prompt for intraday minute trading agents", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=MINUTE_TRADING_SYSTEM_PROMPT, description="System prompt information")

@PROMPT.register_module(force=True)
class IntradayMinuteTradingAgentMessagePrompt(Prompt):
    """Agent message prompt template for intraday minute trading agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='agent_message_prompt', description="The type of the prompt")
    name: str = Field(default="intraday_minute_trading", description="The name of the prompt")
    description: str = Field(default="Agent message prompt for intraday minute trading agents", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=MINUTE_TRADING_MESSAGE_PROMPT, description="Agent message prompt information")
