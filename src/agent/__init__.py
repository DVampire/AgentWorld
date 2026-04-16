"""Agents module for multi-agent system."""

from .tool_calling_agent import ToolCallingAgent
from .planning_agent import PlanningAgent
from .interday_trading_agent import InterdayTradingAgent
from .intraday_trading_agent import IntradayTradingAgent
from .simple_chat_agent import SimpleChatAgent
from .debate_manager import DebateManagerAgent
from .operator_browser_agent import OperatorBrowserAgent
from .mobile_agent import MobileAgent
from .anthropic_mobile_agent import AnthropicMobileAgent
from .online_trading_agent import OnlineTradingAgent
from .offline_trading_agent import OfflineTradingAgent
from .trading_strategy_agent import TradingStrategyAgent
from .deep_analyzer import DeepAnalyzerAgent
from .deep_analyzer_light import DeepAnalyzerLightAgent
from .deep_researcher import DeepResearcherAgent
from .deep_researcher_light import DeepResearcherLightAgent
from .opencode_agent import OpencodeAgent
from .esg_agent import ESGAgent
from .server import agent_manager


__all__ = [
    "ToolCallingAgent",
    "PlanningAgent",
    "InterdayTradingAgent",
    "IntradayTradingAgent",
    "SimpleChatAgent",
    "DebateManagerAgent",
    "OperatorBrowserAgent",
    "MobileAgent",
    "AnthropicMobileAgent",
    "OnlineTradingAgent",
    "OfflineTradingAgent",
    "TradingStrategyAgent",
    "DeepAnalyzerAgent",
    "DeepAnalyzerLightAgent",
    "DeepResearcherAgent",
    "DeepResearcherLightAgent",
    "OpencodeAgent",
    "TradingBenchmarkAgent",
    "TradingSignalAgent",
    "TradingSignalEvaluationAgent",
    "TradingStrategyEvaluationAgent",
    "ESGAgent",
    "agent_manager",
]
