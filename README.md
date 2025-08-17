# AgentWorld

A flexible **asynchronous** multi-agent system built with LangGraph, supporting custom prompt templates, multiple language models, and comprehensive tool management.

## Features

- **🔄 Asynchronous Architecture**: Full async/await support for better performance and concurrency
- **🤖 Multi-Agent Coordination**: Coordinate multiple specialized agents using LangGraph
- **📝 Prompt Management System**: Centralized prompt template management with PromptManager
- **🧠 Model Management System**: Unified management of OpenAI and Anthropic models with ModelManager
- **🔧 Tool Management System**: Comprehensive tool management with ToolManager
- **🛣️ Flexible Routing**: Multiple routing strategies (keyword-based, round-robin, LLM-based)
- **⚡ Concurrent Processing**: Process multiple messages and tools concurrently
- **🔌 Extensible Design**: Easy to add new agents, tools, models, and routing strategies

## Architecture

```
AgentWorld/
├── src/
│   ├── agents/
│   │   ├── base_agent.py      # Async base agent class
│   │   ├── simple_agent.py    # Async simple agent implementation
│   │   ├── multi_agent_system.py  # Async multi-agent coordination
│   │   ├── prompts/
│   │   │   ├── prompt_manager.py  # Centralized prompt management
│   │   │   └── templates/
│   │   │       ├── base_templates.py      # Basic prompt templates
│   │   │       ├── specialized_templates.py  # Specialized agent templates
│   │   │       └── tool_templates.py      # Tool-aware templates
│   │   └── __init__.py
│   ├── models/
│   │   ├── model_manager.py   # Model management system
│   │   ├── base_model.py      # Base model class
│   │   ├── openai_model.py    # OpenAI async model
│   │   ├── anthropic_model.py # Anthropic async model
│   │   └── __init__.py
│   ├── tools/
│   │   ├── tool_manager.py    # Tool management system
│   │   ├── custom_tools.py    # Async custom LangChain tools
│   │   ├── mcp_tools.py       # Async MCP tool integration
│   │   └── __init__.py
│   ├── config/                # Configuration management
│   └── utils/                 # Utility functions
├── configs/                   # Configuration files
├── examples/                  # Usage examples
└── requirements.txt
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AgentWorld
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
```bash
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"
```

### Basic Usage

```python
import asyncio
from src.agents import SimpleAgent, MultiAgentSystem, keyword_based_routing

async def main():
    # Create agent with ModelManager and ToolManager
    agent = SimpleAgent(
        name="assistant",
        model_name="gpt-4",  # Use model from ModelManager
        prompt_name="researcher",  # Use prompt from PromptManager
        tools=["get_current_time", "calculate"]  # Use tools from ToolManager
    )
    
    # Create multi-agent system
    mas = MultiAgentSystem()
    mas.add_agent(agent)
    mas.set_routing_function(keyword_based_routing)
    
    # Process a message asynchronously
    result = await mas.process_message("What time is it?")
    print(result["messages"][-1].content)

# Run the async function
asyncio.run(main())
```

## Model Management System

### ModelManager

The `ModelManager` provides centralized management of all language models:

```python
from src.models import ModelManager, OpenAIAsyncModel, AnthropicAsyncModel

# Initialize ModelManager
model_manager = ModelManager()

# List all available models
models = model_manager.list_models()
print(f"Available models: {models}")

# Get a specific model
model = model_manager.get_model("gpt-4")

# Create new models
custom_model = model_manager.create_openai_model(
    "custom-gpt4",
    model_name="gpt-4",
    temperature=0.5,
    max_tokens=1000
)

claude_model = model_manager.create_anthropic_model(
    "custom-claude",
    model_name="claude-3-sonnet-20240229",
    temperature=0.7
)
```

### Available Models

#### OpenAI Models
- `gpt-4`: GPT-4 model
- `gpt-3.5-turbo`: GPT-3.5 Turbo model
- `gpt-4-turbo`: GPT-4 Turbo model

#### Anthropic Models
- `claude-3-sonnet`: Claude 3 Sonnet model
- `claude-3-opus`: Claude 3 Opus model
- `claude-3-haiku`: Claude 3 Haiku model
- `claude-2`: Claude 2 model

### Using Models with Agents

```python
# Create agent with specific model
agent = SimpleAgent(
    name="researcher",
    model_name="gpt-4",  # Use OpenAI model
    prompt_name="researcher",
    tools=["search_web"]
)

# Create agent with Anthropic model
agent = SimpleAgent(
    name="writer",
    model_name="claude-3-sonnet",  # Use Anthropic model
    prompt_name="writer",
    tools=["file_operations"]
)

# Change model dynamically
agent.change_model("gpt-3.5-turbo")

# Get model information
model_info = agent.get_model_info()
print(f"Current model: {model_info['model_name']}")
```

## Tool Management System

### ToolManager

The `ToolManager` provides comprehensive management of all tools:

```python
from src.tools import ToolManager

# Initialize ToolManager
tool_manager = ToolManager()

# List all available tools
tools = tool_manager.list_tools()
print(f"Available tools: {tools}")

# Get a specific tool
tool = tool_manager.get_tool("get_current_time")

# Execute tools asynchronously
result = await tool_manager.execute_tool("calculate", "2 + 2")

# Execute multiple tools concurrently
tool_calls = [
    {"name": "get_current_time", "args": [], "kwargs": {}},
    {"name": "calculate", "args": ["3 * 4"], "kwargs": {}},
]
results = await tool_manager.execute_multiple_tools(tool_calls)
```

### Available Tools

#### Custom Tools
- `get_current_time`: Get current time
- `search_web`: Web search functionality
- `calculate`: Mathematical calculations
- `weather_lookup`: Weather information
- `file_operations`: File system operations
- `async_web_request`: Async HTTP requests
- `batch_calculation`: Batch mathematical operations

#### MCP Tools
- MCP tool integration for external services
- File system operations via MCP
- Database operations via MCP

### Using Tools with Agents

```python
# Create agent with specific tools
agent = SimpleAgent(
    name="researcher",
    model_name="gpt-4",
    prompt_name="researcher",
    tools=["get_current_time", "search_web"]  # Use tools by name
)

# Add tools dynamically
agent.add_tool("calculate")
agent.add_tool("file_operations")

# Remove tools
agent.remove_tool("get_current_time")

# Get available tools
available_tools = agent.list_available_tools()
```

## Prompt Management System

### PromptManager

The `PromptManager` provides centralized management of all prompt templates:

```python
from src.agents import PromptManager

# Initialize PromptManager
prompt_manager = PromptManager()

# List all available templates
templates = prompt_manager.list_templates()
print(f"Available templates: {templates}")

# Get a specific template
template = prompt_manager.get_template("researcher")

# Add a custom template
prompt_manager.add_template("custom_agent", {
    "template": "You are a custom agent. Human: {input} Agent:",
    "input_variables": ["input"],
    "description": "Custom agent prompt",
    "agent_type": "custom",
    "specialization": "general"
})
```

### Available Prompt Templates

#### Base Templates
- `default`: General assistant prompt
- `general`: General-purpose assistant
- `minimal`: Minimal prompt template
- `conversational`: Natural conversation style

#### Specialized Templates
- `researcher`: Research assistant
- `writer`: Professional writer
- `coder`: Software developer
- `planner`: Project planner
- `analyst`: Data analyst
- `creative`: Creative assistant

#### Tool-Aware Templates
- `tool_assistant`: General assistant with tool access
- `tool_specialist`: Specialized tool usage expert
- `mcp_tool`: MCP tool integration specialist
- `file_operator`: File system operations
- `web_operator`: Web operations specialist

### Using Prompt Templates with Agents

```python
# Create agent with specific prompt template
agent = SimpleAgent(
    name="researcher",
    model_name="gpt-4",
    prompt_name="researcher",  # Use predefined template
    tools=["search_web"]
)

# Switch prompt template dynamically
agent.change_prompt_template("writer")

# Get prompt information
prompt_info = agent.get_prompt_info()
print(f"Current prompt: {prompt_info['specialization']}")
```

## Advanced Usage

### Multi-Agent System with Different Models

```python
from src.agents import SimpleAgent, MultiAgentSystem, keyword_based_routing

# Create multi-agent system with diverse models
mas = MultiAgentSystem(name="diverse_system")

# Add agents with different models
agent_configs = [
    ("researcher", "gpt-4", "researcher", ["get_current_time", "search_web"]),
    ("writer", "claude-3-sonnet", "writer", ["file_operations"]),
    ("coder", "gpt-4-turbo", "coder", ["calculate", "file_operations"]),
    ("analyst", "gpt-3.5-turbo", "analyst", ["calculate", "batch_calculation"]),
]

for agent_name, model_name, prompt_name, tool_names in agent_configs:
    agent = SimpleAgent(
        name=agent_name,
        model_name=model_name,
        prompt_name=prompt_name,
        tools=tool_names
    )
    mas.add_agent(agent)

mas.set_routing_function(keyword_based_routing)

# Process messages
result = await mas.process_message("Research the latest AI developments")
```

### Concurrent Tool Execution

```python
from src.tools import ToolManager

tool_manager = ToolManager()

# Execute multiple tools concurrently
tool_calls = [
    {"name": "get_current_time", "args": [], "kwargs": {}},
    {"name": "calculate", "args": ["2 + 2"], "kwargs": {}},
    {"name": "weather_lookup", "args": ["New York"], "kwargs": {}},
]

results = await tool_manager.execute_multiple_tools(tool_calls)
for i, result in enumerate(results):
    print(f"Tool {i+1} result: {result}")
```

### Dynamic Configuration Changes

```python
# Change model dynamically
agent.change_model("claude-3-opus")

# Change prompt template dynamically
agent.change_prompt_template("creative")

# Add new tools dynamically
agent.add_tool("weather_lookup")

# Get comprehensive agent information
agent_info = agent.get_agent_info()
print(f"Agent info: {agent_info}")
```

## Async Features

### Concurrent Message Processing
```python
# Process multiple messages concurrently
messages = ["What time is it?", "Calculate 2+2", "Write a story"]
results = await mas.process_messages_concurrently(messages)
```

### Concurrent Tool Execution
```python
# Execute multiple tools concurrently
tool_calls = [
    {"name": "get_current_time", "args": [], "kwargs": {}},
    {"name": "calculate", "args": ["2 + 2"], "kwargs": {}},
    {"name": "weather_lookup", "args": ["New York"], "kwargs": {}},
]

results = await tool_manager.execute_multiple_tools(tool_calls)
```

### Async Model Invocation
```python
# Invoke models asynchronously
model = model_manager.get_model("gpt-4")
response = await model.ainvoke("Hello, how are you?")

# Generate multiple responses
messages = [HumanMessage(content="Hello"), HumanMessage(content="How are you?")]
responses = await model.agenerate(messages)
```

## Agent Types

### SimpleAgent (Async)
A basic async agent implementation that supports:
- Model management via ModelManager
- Prompt templates via PromptManager
- Tool management via ToolManager
- Async tool usage (LangChain and MCP tools)
- State management
- Graph-based workflow
- Dynamic model, prompt, and tool changes

### MultiAgentSystem (Async)
Coordinates multiple agents with:
- Intelligent routing between agents
- Conversation state management
- Flexible workflow graphs
- Concurrent message processing
- Support for diverse models and tools

## Tool Integration

### Async Custom Tools
```python
from src.tools import ToolManager

tool_manager = ToolManager()
available_tools = tool_manager.list_tools()
# Includes: get_current_time, search_web, calculate, weather_lookup, 
#          file_operations, async_web_request, batch_calculation
```

### Async MCP Tools
```python
from src.tools import ToolManager

tool_manager = ToolManager()

# Register MCP server
tool_manager.register_mcp_server("file_server", {
    "host": "localhost",
    "port": 8000
})

# Add MCP tool
tool = tool_manager.add_mcp_tool("file_server", "file_operations", {
    "name": "file_operations",
    "description": "File system operations"
})
```

## Routing Strategies

### Keyword-Based Routing
Routes messages based on keyword matching:
```python
from src.agents import keyword_based_routing

mas.set_routing_function(keyword_based_routing)
```

### Round-Robin Routing
Routes messages in a round-robin fashion:
```python
from src.agents import round_robin_routing

mas.set_routing_function(round_robin_routing)
```

### LLM-Based Routing (Async)
Uses an LLM to intelligently route messages:
```python
from src.agents import llm_based_routing

routing_func = llm_based_routing(llm, agents)
mas.set_routing_function(routing_func)
```

## Configuration

Configuration files are stored in `configs/`:

```python
# configs/multi_agent.py
agents = {
    "researcher": {
        "name": "researcher",
        "type": "SimpleAgent",
        "model_name": "gpt-4",  # Use ModelManager
        "prompt_name": "researcher",  # Use PromptManager
        "tools": ["search_web", "get_current_time"],  # Use ToolManager
        "mcp_tools": []
    }
}
```

## Running Demos

### ModelManager & ToolManager Demo
```bash
python run_manager_demo.py
```

### PromptManager Demo
```bash
python run_prompt_demo.py
```

### Full Async Demo
```bash
python examples/model_tool_manager_example.py
```

## Performance Benefits

The async architecture provides several performance benefits:

1. **Concurrent Processing**: Multiple agents can process messages simultaneously
2. **Non-blocking I/O**: Tools and API calls don't block the event loop
3. **Better Resource Utilization**: Efficient use of system resources
4. **Scalability**: Can handle more concurrent requests
5. **Responsiveness**: System remains responsive during long operations
6. **Model Flexibility**: Easy switching between different language models
7. **Tool Efficiency**: Concurrent tool execution and management

## Customization

### Adding New Async Models
1. Inherit from `BaseModel`
2. Implement required async methods (`ainvoke`, `agenerate`)
3. Register with `ModelManager`

### Adding New Async Tools
1. Create async LangChain tool using `@tool` decorator
2. Add to `ToolManager`
3. Assign to agents as needed

### Adding New Prompt Templates
1. Create Python file in `src/agents/prompts/templates/`
2. Define `PROMPT_TEMPLATES` dictionary or individual template variables
3. Templates are automatically loaded by PromptManager

### Adding New Routing Strategies
1. Create async routing function that takes `(message, state)` and returns agent name
2. Set as routing function in `MultiAgentSystem`

## Testing

Run the async tests:
```bash
pytest tests/test_async_multi_agent.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes (prefer async implementations)
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.