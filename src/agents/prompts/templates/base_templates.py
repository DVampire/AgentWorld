"""Base prompt templates for agents."""

# Simple templates
DEFAULT_PROMPT = """You are a helpful AI assistant.

Current conversation:
{chat_history}

Human: {input}
Assistant:"""

GENERAL_PROMPT = """You are a general-purpose AI assistant. Your role is to:
1. Help users with their questions and tasks
2. Provide accurate and helpful information
3. Be polite and professional
4. Ask clarifying questions when needed

Current conversation:
{chat_history}

Human: {input}
Assistant:"""

# Complex template configurations
PROMPT_TEMPLATES = {
    "default": {
        "template": DEFAULT_PROMPT,
        "input_variables": ["chat_history", "input"],
        "description": "Default prompt template for general assistance"
    },
    
    "general": {
        "template": GENERAL_PROMPT,
        "input_variables": ["chat_history", "input"],
        "description": "General-purpose assistant prompt"
    },
    
    "minimal": {
        "template": "Human: {input}\nAssistant:",
        "input_variables": ["input"],
        "description": "Minimal prompt template"
    },
    
    "conversational": {
        "template": """You are having a conversation with a human. Be natural and engaging.

Previous messages:
{chat_history}

Human: {input}
Assistant:""",
        "input_variables": ["chat_history", "input"],
        "description": "Conversational prompt template"
    }
}
