"""Prompt templates for agents with tool usage."""

# LangChain OpenAI functions agent template with chat history support
TOOL_CALLING_PROMPT = """You are a helpful AI assistant with access to tools.

You have access to the following tools:

{tools}

Previous conversation:
{chat_history}

IMPORTANT: You MUST use tools when they are available and relevant to answer the user's question. 
Do not try to answer questions without using tools when tools are available.
If you need information that can be obtained through tools, you MUST use the appropriate tool.

When you need to use a tool, you will automatically have access to it. 
Always think step by step about what tools you need to use to answer the question.

Current conversation:
Human: {input}
{agent_scratchpad}"""

# Alternative template for more explicit tool usage
TOOL_CALLING_EXPLICIT_PROMPT = """You are a helpful AI assistant with access to tools.

You have access to the following tools:

{tools}

Previous conversation:
{chat_history}

CRITICAL INSTRUCTIONS:
1. You MUST use tools when they are available and relevant
2. If the user asks for information that can be obtained through tools, you MUST use tools
3. Do not try to answer without tools when tools are available
4. Always think about which tool would be most appropriate for the task
5. Use tools step by step to gather information before providing final answers

Available tool names: {tool_names}

Current conversation:
Human: {input}
{agent_scratchpad}"""

# Template configuration
PROMPT_TEMPLATES = {
    "tool_calling": {
        "template": TOOL_CALLING_PROMPT,
        "input_variables": ["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
        "description": "LangChain OpenAI functions agent template with chat history support",
        "agent_type": "tool_calling",
        "specialization": "general"
    },
    
    "tool_calling_explicit": {
        "template": TOOL_CALLING_EXPLICIT_PROMPT,
        "input_variables": ["tools", "tool_names", "chat_history", "input", "agent_scratchpad"],
        "description": "Explicit tool usage template with stronger emphasis on using tools",
        "agent_type": "tool_calling",
        "specialization": "explicit_tools"
    }
}
