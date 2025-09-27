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
- **🔍 Advanced Research Tools**: Multi-round research workflow with DeepResearcherTool
- **🖼️ Multimodal Support**: Image and text analysis capabilities in research tools

## Architecture

```
AgentWorld/
├── src/
│   ├── agents/
│   │   ├── base_agent.py      # Async base agent class
│   │   ├── interactive_agent.py    # Interactive agent implementation
│   │   ├── tool_calling_agent.py   # Tool calling agent implementation
│   │   └── __init__.py
│   ├── models/
│   │   ├── model_manager.py   # Model management system
│   │   ├── message_manager.py # Message management
│   │   └── __init__.py
│   ├── tools/
│   │   ├── tool_manager.py    # Central tool management system
│   │   ├── default_tools/     # Basic utility tools
│   │   │   ├── web_searcher.py    # Web search functionality
│   │   │   ├── web_fetcher.py     # Web content fetching
│   │   │   └── default_tool_set.py # Default tool collection
│   │   ├── environment_tools/ # Environment-specific tools
│   │   │   └── environment_tool_set.py # Environment tool collection
│   │   ├── agent_tools/       # Workflow and agent-specific tools
│   │   │   ├── browser.py         # Browser automation tool
│   │   │   ├── deep_researcher.py # Multi-round research workflow
│   │   │   └── agent_tool_set.py  # Agent tool collection
│   │   ├── mcp_tools/         # MCP protocol tools
│   │   │   └── mcp_tool_set.py    # MCP tool collection
│   │   └── __init__.py
│   ├── prompts/
│   │   ├── prompt_manager.py  # Centralized prompt management
│   │   ├── system_prompt.py   # System prompt templates
│   │   ├── agent_message_prompt.py # Agent message prompts
│   │   └── templates/         # Prompt template files
│   ├── config/                # Configuration management
│   ├── datasets/              # Data loading and processing
│   ├── environments/          # Trading and simulation environments
│   ├── memory/                # Memory management system
│   ├── metric/                # Performance metrics
│   ├── logger/                # Logging system
│   ├── filesystem/            # File system utilities
│   └── utils/                 # Utility functions
├── configs/                   # Configuration files
├── examples/                  # Usage examples
├── datasets/                  # Data files
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

**Important**: The `use_local_proxy` parameter in configuration files determines whether to use local proxy services:
- Set to `False` when using official OpenAI/Anthropic APIs directly
- Set to `True` when using local proxy services (like localhost:8000)

### Basic Usage

```python
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(verbose=True)

# Add project root to path
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.registry import AGENTS
from src.models import model_manager
from src.tools import tool_manager

async def main():
    # Initialize configuration
    config.init_config("configs/tool_calling_agent.py")
    logger.init_logger(config)
    
    # Initialize models and tools
    # Note: use_local_proxy is read from config file
    await model_manager.init_models(use_local_proxy=config.use_local_proxy)
    await tool_manager.init_tools()
    
    # Create agent using registry
    agent = AGENTS.build(config.agent)
    
    # Run the agent
    task = "Search for information about AI"
    await agent.run(task)

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
agent = ToolCallingAgent(
    name="researcher",
    model_name="gpt-4",  # Use OpenAI model
    tools=["web_searcher_tool", "web_fetcher_tool"]
)

# Create agent with Anthropic model
agent = ToolCallingAgent(
    name="writer",
    model_name="claude-3-sonnet",  # Use Anthropic model
    tools=["web_searcher_tool", "web_fetcher_tool"]
)

# Change model dynamically
agent.change_model("gpt-3.5-turbo")

# Get model information
model_info = agent.get_model_info()
print(f"Current model: {model_info['model_name']}")
```

## Tool Management System

### ToolManager

The `ToolManager` provides comprehensive management of all tools organized into specialized categories:

```python
from src.tools import ToolManager

# Initialize ToolManager
tool_manager = ToolManager()

# List all available tools
tools = tool_manager.list_tools()
print(f"Available tools: {tools}")

# Get a specific tool
tool = tool_manager.get_tool("web_searcher_tool")

# Execute tools asynchronously
result = await tool_manager.execute_tool("web_fetcher_tool", "https://example.com")

# Execute multiple tools concurrently
tool_calls = [
    {"name": "web_searcher_tool", "args": ["AI research"], "kwargs": {}},
    {"name": "web_fetcher_tool", "args": ["https://example.com"], "kwargs": {}},
]
results = await tool_manager.execute_multiple_tools(tool_calls)
```

### Tool Categories

#### Default Tools (`default_tools/`)
Basic utility tools for common operations:
- `web_searcher`: Web search using multiple search engines (Firecrawl, Google, DuckDuckGo, Baidu, Bing)
- `web_fetcher`: Fetch and extract content from web pages
- `bash`: Execute bash commands
- `file`: File operations
- `project`: Project management
- `python_interpreter`: Python code execution
- `weather`: Weather information lookup
- `done`: Task completion marker

#### Environment Tools (`environment_tools/`)
Tools specific to trading and simulation environments:
- `environment_tool_set.py`: Collection and management of environment-specific tools

#### Agent Tools (`agent_tools/`)
Advanced workflow and agent-specific tools:
- `browser`: Browser automation for web interactions
- `deep_researcher`: Multi-round research workflow agent that:
  - Generates optimized search queries using LLM
  - Performs multi-round web searches
  - Extracts insights from web content
  - Provides comprehensive research summaries
  - Supports multimodal input (text + image)

#### MCP Tools (`mcp_tools/`)
Model Context Protocol tools for external service integration:
- `mcp_tool_set.py`: Collection and management of MCP protocol tools

### Using Tools with Agents

```python
# Create agent with specific tools
agent = AGENTS.build(dict(
    type="ToolCallingAgent",
    name="researcher",
    model_name="gpt-4",
    tools=["web_searcher", "web_fetcher", "browser", "deep_researcher"]
))

# Tools are configured in the config file and loaded automatically
available_tools = agent.list_available_tools()
print(f"Available tools: {available_tools}")
```

### Deep Researcher Tool Example

```python
from src.tools.agent_tools import DeepResearcherTool

# Initialize the deep researcher tool
researcher = DeepResearcherTool(model_name="o3")

# Perform multi-round research
result = await researcher.arun(
    task="What are the latest developments in quantum computing?",
    image=None,  # Optional: path to image for multimodal analysis
    filter_year=2024  # Optional: filter results by year
)

print(result)  # Comprehensive research summary
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
# Create agent with specific tools
agent = ToolCallingAgent(
    name="researcher",
    model_name="gpt-4",
    tools=["web_searcher_tool", "web_fetcher_tool"]
)

# Add new tools dynamically
agent.add_tool("deep_researcher_tool")

# Get agent information
agent_info = agent.get_agent_info()
print(f"Agent tools: {agent_info['tools']}")
```

## Advanced Usage

### Advanced Tool Usage with Different Models

```python
from src.registry import AGENTS
from src.tools.agent_tools import DeepResearcherTool

# Create specialized agents with different models
researcher_agent = AGENTS.build(dict(
    type="ToolCallingAgent",
    name="researcher",
    model_name="gpt-4",
    tools=["web_searcher", "web_fetcher", "deep_researcher"]
))

writer_agent = AGENTS.build(dict(
    type="ToolCallingAgent",
    name="writer",
    model_name="claude-3-sonnet",
    tools=["web_searcher", "web_fetcher"]
))

# Use deep researcher tool directly
deep_researcher = DeepResearcherTool(model_name="o3")
research_result = await deep_researcher.arun(
    task="Research the latest AI developments",
    filter_year=2024
)

print(f"Research completed: {research_result}")
```

### Concurrent Tool Execution

```python
from src.tools import ToolManager

tool_manager = ToolManager()

# Execute multiple tools concurrently
tool_calls = [
    {"name": "web_searcher_tool", "args": ["AI research"], "kwargs": {}},
    {"name": "web_fetcher_tool", "args": ["https://example.com"], "kwargs": {}},
    {"name": "deep_researcher_tool", "args": ["Quantum computing"], "kwargs": {}},
]

results = await tool_manager.execute_multiple_tools(tool_calls)
for i, result in enumerate(results):
    print(f"Tool {i+1} result: {result}")
```

### Dynamic Configuration Changes

```python
# Change model dynamically
agent.change_model("claude-3-opus")

# Add new tools dynamically
agent.add_tool("deep_researcher_tool")

# Remove tools
agent.remove_tool("web_searcher_tool")

# Get comprehensive agent information
agent_info = agent.get_agent_info()
print(f"Agent info: {agent_info}")
```

## Async Features

### Concurrent Message Processing
```python
# Process multiple messages concurrently
messages = ["Search for AI news", "Fetch content from example.com", "Research quantum computing"]
results = await agent.process_messages_concurrently(messages)
```

### Concurrent Tool Execution
```python
# Execute multiple tools concurrently
tool_calls = [
    {"name": "web_searcher_tool", "args": ["AI research"], "kwargs": {}},
    {"name": "web_fetcher_tool", "args": ["https://example.com"], "kwargs": {}},
    {"name": "deep_researcher_tool", "args": ["Quantum computing"], "kwargs": {}},
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

### InteractiveAgent (Async)
An interactive agent implementation that supports:
- Model management via ModelManager
- Prompt templates via PromptManager
- Tool management via ToolManager
- Async tool usage (LangChain and MCP tools)
- Interactive conversation capabilities
- State management and memory

### ToolCallingAgent (Async)
A specialized agent for tool execution that supports:
- Model management via ModelManager
- Tool management via ToolManager
- Async tool usage with proper error handling
- Structured tool calling and response processing
- Integration with all tool categories (default, environment, agent, MCP)
- Support for complex workflow tools like DeepResearcherTool

## Tool Integration

### Async Custom Tools
```python
from src.tools import ToolManager

tool_manager = ToolManager()
available_tools = tool_manager.list_tools()
# Includes: web_searcher_tool, web_fetcher_tool, 
#          deep_researcher_tool, and MCP tools
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

## Tool Integration Examples

### Web Search and Content Fetching
```python
from src.tools.default_tools import WebSearcherTool, WebFetcherTool

# Initialize tools
searcher = WebSearcherTool()
fetcher = WebFetcherTool()

# Search for information
search_results = await searcher.arun("AI research papers 2024")
print(f"Found {len(search_results)} results")

# Fetch content from a specific URL
content = await fetcher.arun("https://example.com")
print(f"Content length: {len(content)}")
```

### Browser Automation
This feature has been removed. Please use other automation tools as needed.

## Configuration

Configuration files are stored in `configs/`. **Important**: Pay attention to the `use_local_proxy` parameter which affects API connectivity.

### Proxy Configuration

The `use_local_proxy` parameter is crucial for proper API connectivity:

- **`use_local_proxy = False`**: Use official OpenAI/Anthropic APIs directly
  - Requires valid API keys in environment variables
  - Direct connection to OpenAI/Anthropic servers
  - Suitable for production environments

- **`use_local_proxy = True`**: Use local proxy services
  - Requires local proxy server running (e.g., localhost:8000)
  - Useful for development/testing with local models
  - Can help with rate limiting and cost management

### Configuration Files

```python
# configs/tool_calling_agent.py
from mmengine.config import read_base
with read_base():
    from .base import deep_researcher_tool

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

# IMPORTANT: Set use_local_proxy based on your API setup
use_local_proxy = False  # Set to False for official OpenAI/Anthropic APIs
# use_local_proxy = True   # Set to True for local proxy services (localhost:8000)
version = "0.1.0"

agent = dict(
    type = "ToolCallingAgent",
    name = "tool_calling_agent",
    model_name = "gpt-4.1",
    prompt_name = "tool_calling",  # Use explicit tool usage template
    tools = [
        "bash",
        "file",
        "project",
        "python_interpreter",
        "browser",
        "done",
        "weather",
        "web_fetcher",
        "web_searcher",
        "deep_researcher",
    ],
    max_iterations = 10
)
```

### Base Configuration

```python
# configs/base.py
from mmengine.config import read_base
with read_base():
    from .environments.trading_offline import dataset, environment, metric

# Tool-specific configurations
deep_researcher_tool = dict(
    model_name = "o3",  # Research-specific model
)
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
2. Implement required async methods (`ainvoke`)
3. Register with `ModelManager`

### Adding New Async Tools
1. Inherit from `langchain.tools.BaseTool`
2. Implement required async methods (`_arun`, `_run`)
3. Add to appropriate tool set (`default_tools`, `environment_tools`, `agent_tools`, or `mcp_tools`)
4. Register with `ToolManager`

### Adding New Prompt Templates
1. Create Python file in `src/agents/prompts/templates/`
2. Define `PROMPT_TEMPLATES` dictionary or individual template variables
3. Templates are automatically loaded by PromptManager

### Adding New Routing Strategies
1. Create async routing function that takes `(message, state)` and returns agent name
2. Set as routing function in your custom agent system

## Testing

Run the tests:
```bash
# Test agents
pytest tests/test_agent.py -v

# Test tools
# Browser tool tests have been removed

# Test models
pytest tests/test_models.py -v

# Test prompt management
pytest tests/test_prompt_management.py -v
```

## Troubleshooting

### Common Issues

#### API Connection Problems
- **Error**: "Failed to connect to OpenAI API"
  - **Solution**: Check if `use_local_proxy` is set correctly in your config file
  - For official APIs: `use_local_proxy = False`
  - For local proxy: `use_local_proxy = True` (ensure proxy server is running)

#### Model Initialization Failures
- **Error**: "Model initialization failed"
  - **Solution**: Verify your API keys are set correctly in environment variables
  - Check if the model name exists in your ModelManager
  - Ensure network connectivity to API endpoints

#### Tool Loading Issues
- **Error**: "Tool not found" or "Tool initialization failed"
  - **Solution**: Check if the tool is properly registered in the appropriate tool set
  - Verify tool dependencies are installed
  - Check tool configuration in base.py

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes (prefer async implementations)
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.