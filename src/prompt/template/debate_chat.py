from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict
from pydantic import Field, ConfigDict

AGENT_PROFILE = """
You are a knowledgeable AI assistant participating in a multi-agent debate. You are designed to engage in thoughtful, substantive discussions and provide well-reasoned arguments.
"""

PERSONALITY = """
<personality>
- Be analytical and evidence-based in your responses
- Present clear, logical arguments with supporting points
- Challenge ideas constructively when appropriate
- Acknowledge valid points from other participants
- Stay focused on the debate topic
- Be respectful but assertive in your positions
</personality>
"""

DEBATE_GUIDELINES = """
<debate_guidelines>
- Build upon previous arguments in the conversation
- Provide specific evidence, examples, or data when possible
- Address counter-arguments directly
- Ask probing questions to deepen the discussion
- Avoid repetitive or circular arguments
- Stay on topic and contribute meaningfully
</debate_guidelines>
"""

RESPONSE_FORMAT = """
<response_format>
- Write in a clear, professional debate style
- Structure arguments logically with clear points
- Use evidence and examples to support your position
- Engage directly with what others have said
- End with questions or challenges that advance the debate
</response_format>
"""

SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ personality }}
{{ debate_guidelines }}
{{ response_format }}
"""

# Agent message (dynamic context) - using Jinja2 syntax
AGENT_MESSAGE_PROMPT_TEMPLATE = """
{{ agent_context }}
{{ environment_context }}
{{ tool_context }}
{{ examples }}
"""

SYSTEM_PROMPT = {
    "name": "debate_chat_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for debate chat agents - analytical debate personality",
    "template": SYSTEM_PROMPT_TEMPLATE,
    "variables": [
        {
            "name": "agent_profile",
            "type": "system_prompt_module",
            "description": "Describes the debate agent's core identity and capabilities for engaging in multi-agent debates.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_PROFILE
        },
        {
            "name": "personality",
            "type": "system_prompt_module",
            "description": "Defines the analytical and evidence-based personality traits for debate participation.",
            "require_grad": False,
            "template": None,
            "variables": PERSONALITY
        },
        {
            "name": "debate_guidelines",
            "type": "system_prompt_module",
            "description": "Provides guidelines for engaging in substantive debates and building upon previous arguments.",
            "require_grad": False,
            "template": None,
            "variables": DEBATE_GUIDELINES
        },
        {
            "name": "response_format",
            "type": "system_prompt_module",
            "description": "Specifies the format and style requirements for debate responses.",
            "require_grad": False,
            "template": None,
            "variables": RESPONSE_FORMAT
        }
    ]
}

AGENT_MESSAGE_PROMPT = {
    "name": "debate_chat_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for debate chat agents (debate context)",
    "template": AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": [
        {
            "name": "agent_context",
            "type": "agent_message_prompt_module",
            "description": "Describes the debate agent's current state, including current debate topic, conversation history, and plans.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "environment_context",
            "type": "agent_message_prompt_module",
            "description": "Describes the debate environment, including current time and conversation context.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "tool_context",
            "type": "agent_message_prompt_module",
            "description": "Describes available tools and their usage conditions for the debate agent.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        {
            "name": "examples",
            "type": "agent_message_prompt_module",
            "description": "Contains few-shot examples of good debate patterns and argumentation strategies.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
    ],
}

@PROMPT.register_module(force=True)
class DebateChatPrompt(Prompt):
    """Prompt template for debate chat agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="debate_chat", description="The name of the prompt")
    description: str = Field(default="Prompt for debate chat agents", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    @property
    def system_prompt(self) -> Dict[str, Any]:
        return SYSTEM_PROMPT
    
    @property
    def agent_message_prompt(self) -> Dict[str, Any]:
        return AGENT_MESSAGE_PROMPT
