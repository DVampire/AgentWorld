from src.registry import PROMPT
from src.prompt.types import Prompt
from typing import Any, Dict, Literal
from pydantic import Field, ConfigDict

AGENT_PROFILE = """
You are a helpful AI assistant that can have natural conversations with humans. You are designed to be friendly, informative, and engaging.
"""

PERSONALITY = """
<personality>
- Be conversational and approachable
- Show genuine interest in the user's questions and concerns
- Provide helpful, accurate information
- Be concise but thorough in your responses
- Ask follow-up questions when appropriate
- Admit when you don't know something
</personality>
"""

CONVERSATION_GUIDELINES = """
<conversation_guidelines>
- Respond naturally as if talking to a friend
- Keep responses conversational but informative
- Use appropriate tone based on the user's message
- Be empathetic and understanding
- Provide examples or analogies when helpful
- Encourage further discussion when relevant
</conversation_guidelines>
"""

RESPONSE_FORMAT = """
<response_format>
- Write in a natural, conversational style
- Use appropriate punctuation and formatting
- Keep responses focused and relevant
- End with questions or prompts for continued conversation when appropriate
</response_format>
"""

SYSTEM_PROMPT_TEMPLATE = """
{{ agent_profile }}
{{ personality }}
{{ conversation_guidelines }}
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
    "name": "simple_chat_system_prompt",
    "type": "system_prompt",
    "description": "System prompt for simple chat agents - conversational personality",
    "template": SYSTEM_PROMPT_TEMPLATE,
    "variables": {
        "agent_profile": {
            "name": "agent_profile",
            "type": "system_prompt",
            "description": "Describes the chat agent's core identity and capabilities for natural conversations.",
            "require_grad": False,
            "template": None,
            "variables": AGENT_PROFILE
        },
        "personality": {
            "name": "personality",
            "type": "system_prompt",
            "description": "Defines the conversational and approachable personality traits for chat interactions.",
            "require_grad": False,
            "template": None,
            "variables": PERSONALITY
        },
        "conversation_guidelines": {
            "name": "conversation_guidelines",
            "type": "system_prompt",
            "description": "Provides guidelines for engaging in natural, friendly conversations.",
            "require_grad": False,
            "template": None,
            "variables": CONVERSATION_GUIDELINES
        },
        "response_format": {
            "name": "response_format",
            "type": "system_prompt",
            "description": "Specifies the format and style requirements for conversational responses.",
            "require_grad": False,
            "template": None,
            "variables": RESPONSE_FORMAT
        }
    }
}

AGENT_MESSAGE_PROMPT = {
    "name": "simple_chat_agent_message_prompt",
    "type": "agent_message_prompt",
    "description": "Agent message for simple chat agents (conversation context)",
    "require_grad": False,
    "template": AGENT_MESSAGE_PROMPT_TEMPLATE,
    "variables": {
        "agent_context": {
            "name": "agent_context",
            "type": "agent_message_prompt",
            "description": "Describes the chat agent's current state, including current conversation topic, conversation history, and plans.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "environment_context": {
            "name": "environment_context",
            "type": "agent_message_prompt",
            "description": "Describes the conversation environment, including current time and conversation context.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "tool_context": {
            "name": "tool_context",
            "type": "agent_message_prompt",
            "description": "Describes available tools and their usage conditions for the chat agent.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
        "examples": {
            "name": "examples",
            "type": "agent_message_prompt",
            "description": "Contains few-shot examples of good conversation patterns and response strategies.",
            "require_grad": False,
            "template": None,
            "variables": None
        },
    },
}

@PROMPT.register_module(force=True)
class SimpleChatSystemPrompt(Prompt):
    """System prompt template for simple chat agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='system_prompt', description="The type of the prompt")
    name: str = Field(default="simple_chat", description="The name of the prompt")
    description: str = Field(default="System prompt for simple chat agents", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=SYSTEM_PROMPT, description="System prompt information")

@PROMPT.register_module(force=True)
class SimpleChatAgentMessagePrompt(Prompt):
    """Agent message prompt template for simple chat agents."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    type: str = Field(default='agent_message_prompt', description="The type of the prompt")
    name: str = Field(default="simple_chat", description="The name of the prompt")
    description: str = Field(default="Agent message prompt for simple chat agents", description="The description of the prompt")
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the prompt")
    
    prompt_config: Dict[str, Any] = Field(default=AGENT_MESSAGE_PROMPT, description="Agent message prompt information")
