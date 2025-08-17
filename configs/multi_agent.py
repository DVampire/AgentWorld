# Multi-agent system configuration

# System settings
workdir = "workdir"
tag = "multi_agent"
exp_path = f"{workdir}/{tag}"
log_path = "multi_agent.log"

# Model settings
model_id = "gpt-4"
version = "0.1.0"

# Agent configurations
agents = {
    "researcher": {
        "name": "researcher",
        "type": "SimpleAgent",
        "prompt_template": """You are a research assistant. Your role is to:
1. Analyze and research topics thoroughly
2. Find relevant information and sources
3. Provide detailed, well-structured research findings
4. Ask clarifying questions when needed

Current conversation:
{chat_history}

Human: {input}
Researcher:""",
        "tools": ["search_web", "get_current_time"],
        "mcp_tools": []
    },
    
    "writer": {
        "name": "writer", 
        "type": "SimpleAgent",
        "prompt_template": """You are a professional writer. Your role is to:
1. Create well-written, engaging content
2. Adapt writing style to different audiences
3. Structure content logically and clearly
4. Edit and improve existing content

Current conversation:
{chat_history}

Human: {input}
Writer:""",
        "tools": ["file_operations"],
        "mcp_tools": []
    },
    
    "coder": {
        "name": "coder",
        "type": "SimpleAgent", 
        "prompt_template": """You are a software developer. Your role is to:
1. Write clean, efficient code
2. Debug and fix issues
3. Explain technical concepts clearly
4. Suggest best practices and improvements

Current conversation:
{chat_history}

Human: {input}
Coder:""",
        "tools": ["calculate", "file_operations"],
        "mcp_tools": []
    },
    
    "planner": {
        "name": "planner",
        "type": "SimpleAgent",
        "prompt_template": """You are a project planner. Your role is to:
1. Break down complex tasks into manageable steps
2. Create timelines and schedules
3. Identify dependencies and risks
4. Coordinate between different team members

Current conversation:
{chat_history}

Human: {input}
Planner:""",
        "tools": ["get_current_time", "calculate"],
        "mcp_tools": []
    }
}

# Routing configuration
routing = {
    "type": "keyword_based",  # Options: "round_robin", "keyword_based", "llm_based"
    "keywords": {
        "researcher": ["research", "study", "analysis", "investigate", "find", "search"],
        "writer": ["write", "compose", "draft", "create", "generate", "edit"],
        "coder": ["code", "program", "develop", "implement", "debug", "software"],
        "planner": ["plan", "organize", "schedule", "coordinate", "manage", "timeline"]
    }
}

# MCP tools configuration
mcp_tools = {
    "file_system": {
        "name": "file_system",
        "description": "File system operations",
        "server": "file_system_server",
        "methods": ["read", "write", "list", "delete"]
    },
    "database": {
        "name": "database", 
        "description": "Database operations",
        "server": "database_server",
        "methods": ["query", "insert", "update", "delete"]
    }
}
