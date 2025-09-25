# AgentOrchestra

A comprehensive multi-agent system implementing the **TEA Protocol** (Tool-Environment-Agent Protocol), built with LangChain and featuring advanced tool management, multiple language models, and specialized environments for various use cases.

## ✨ Key Features

- **🔧 TEA Protocol Implementation**: Complete implementation of Tool-Environment-Agent Protocol with seamless interoperability
- **🔄 Full Async Architecture**: Complete async/await implementation for optimal performance
- **🤖 Multi-Agent Coordination**: Sophisticated agent orchestration with protocol-based communication
- **🧠 Unified Model Management**: Support for OpenAI, Anthropic, Google, and custom models with local proxy support
- **🛠️ Comprehensive Tool System**: 50+ tools or actions across multiple categories (web, file, browser, research, MCPactions
- **🌍 Rich Environment Support**: File system, GitHub, database, trading, browser automation, vector search
- **📝 Advanced Prompt Management**: Centralized template system with specialized prompts
- **🔍 Deep Research Capabilities**: Multi-round research workflows with multimodal support
- **⚡ High Performance**: Concurrent processing and optimized resource utilization
- **🔌 Extensible Design**: Easy integration of new agents, tools, and environments
- **🔄 Protocol Transformations**: Dynamic resource orchestration with A2T, T2A, E2T, T2E, A2E, E2A transformations

## 🔧 The TEA Protocol

AgentOrchestra implements the **TEA Protocol** (Tool-Environment-Agent Protocol), a unified framework for managing tools, environments, and agents with seamless interoperability.

### Core Components

- **🛠️ Tool Context Protocol (TCP)**: Unified tool management with semantic relationships and context-aware execution
- **🌍 Environment Context Protocol (ECP)**: Standardized environment interfaces across diverse domains  
- **🤖 Agent Context Protocol (ACP)**: Comprehensive agent orchestration with persistent state management

### Protocol Transformations

The TEA Protocol supports six key transformations for dynamic resource orchestration:

- **A2T**: Agent-to-Tool (encapsulate agent capabilities as tools)
- **T2A**: Tool-to-Agent (designate tools as agent actuators)
- **E2T**: Environment-to-Tool (convert environment actions to tool interfaces)
- **T2E**: Tool-to-Environment (elevate tool sets to environment abstractions)
- **A2E**: Agent-to-Environment (encapsulate agents as interactive environments)
- **E2A**: Environment-to-Agent (infuse reasoning into environment dynamics)

This enables seamless transitions between computational entities, supporting adaptive architectures that reconfigure components based on task requirements.

## 🏗️ Architecture Overview

```
AgentOrchestra/
├── src/
│   ├── agents/                 # Agent implementations (TEA Protocol ACP)
│   │   ├── tool_calling_agent.py    # Tool-calling agent with async support
│   │   ├── simple_chat_agent.py    # Conversational agent
│   │   ├── debate_manager.py        # Multi-agent debate coordination
│   │   ├── finagent.py             # Financial analysis agent
│   │   ├── prompts/                 # Prompt templates and management
│   │   └── protocol/               # Agent Context Protocol (ACP)
│   ├── environments/           # Specialized environments (TEA Protocol ECP)
│   │   ├── file_system_environment.py    # File operations
│   │   ├── github_environment.py         # GitHub integration
│   │   ├── database_environment.py       # Database operations
│   │   ├── trading_offline_environment.py # Financial trading
│   │   ├── playwright_environment.py    # Browser automation
│   │   ├── faiss_environment.py         # Vector search
│   │   └── protocol/                    # Environment Context Protocol (ECP)
│   ├── tools/                  # Comprehensive tool system (TEA Protocol TCP)
│   │   ├── default_tools/      # Core utilities (web, file, bash, etc.)
│   │   ├── workflow_tools/      # Advanced workflows (research, analysis)
│   │   ├── mcp_tools/          # Model Context Protocol tools
│   │   └── protocol/           # Tool Context Protocol (TCP)
│   ├── infrastructures/        # Core infrastructure
│   │   ├── models/             # Model management (OpenAI, Anthropic, Google)
│   │   └── memory/            # Memory and state management
│   ├── transformation/         # Protocol transformations (A2T, T2A, E2T, T2E, A2E, E2A)
│   ├── supports/              # Supporting utilities
│   │   ├── datasets/          # Data processing
│   │   ├── metric/            # Performance metrics
│   │   └── calen/             # Calendar utilities
│   └── utils/                 # Common utilities
├── configs/                   # Configuration files
├── examples/                  # Usage examples and demos
├── tests/                     # Comprehensive test suite
└── libs/                     # External libraries (browser-use, langchain-community)
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git
- API keys for your preferred LLM providers

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd AgentOrchestra
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
```bash
# Required API keys
export OPENAI_API_KEY="your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"

# Optional API keys for additional features
export GOOGLE_API_KEY="your-google-key-here"
export GITHUB_TOKEN="your-github-token-here"
export BRAVE_SEARCH_API_KEY="your-brave-key-here"
export FIRECRAWL_API_KEY="your-firecrawl-key-here"
export AO_API_KEY="your-ao-key-here"  # For local proxy support
```

### 🔧 Configuration

The `use_local_proxy` parameter in configuration files controls API connectivity:

- **`use_local_proxy = False`**: Use official APIs directly (recommended for production)
- **`use_local_proxy = True`**: Use local proxy services (useful for development/testing)

### 🎯 Basic Usage

**Simple Chat Agent:**
```python
import asyncio
from pathlib import Path
import sys

# Add project root to path
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.infrastructures.models import model_manager
from src.agents import acp

async def main():
    # Initialize configuration
    config.init_config("configs/simple_chat_agent.py")
    logger.init_logger(config)
    
    # Initialize models and agents
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    await acp.initialize(config.agent_names)
    
    # Run simple chat
    result = await acp.ainvoke(
        name="simple_chat",
        input={"task": "Hello, how are you?"}
    )
    print(result)

asyncio.run(main())
```

**Tool-Calling Agent:**
```python
import asyncio
from pathlib import Path
import sys

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.infrastructures.models import model_manager
from src.tools import tcp
from src.agents import acp

async def main():
    # Initialize with tool-calling configuration
    config.init_config("configs/tool_calling_agent.py")
    logger.init_logger(config)
    
    # Initialize all systems
    await model_manager.initialize(use_local_proxy=config.use_local_proxy)
    await tcp.initialize(config.tool_names)
    await acp.initialize(config.agent_names)
    
    # Run tool-calling agent
    result = await acp.ainvoke(
        name="tool_calling",
        input={"task": "Search for the latest AI news and summarize it"}
    )
    print(result)

asyncio.run(main())
```

## 🛠️ Available Tools & Environments

### Core Tools (Default Tools)
- **Web Operations**: `web_searcher`, `web_fetcher` - Multi-engine search and content extraction
- **File Operations**: `file`, `bash`, `python_interpreter` - File system and code execution
- **Project Management**: `project`, `todo`, `done` - Task and project organization
- **Utilities**: `weather`, `mdify` - Weather info and document conversion

### Advanced Workflow Tools
- **Deep Researcher**: Multi-round research with LLM-optimized queries and multimodal support
- **Deep Analyzer**: Advanced content analysis and insights
- **Browser Tool**: Full browser control with Playwright integration

### MCP Tools (Model Context Protocol)
- **MCP Client/Server**: Integration with external MCP services
- **File Operations**: MCP-based file system operations
- **External Services**: Integration with various external services via MCP

### Specialized Environments
- **File System Environment**: Complete file operations and management with path policies
- **GitHub Environment**: Repository management, cloning, commits, and collaboration
- **Database Environment**: SQL operations with multiple database support
- **Trading Environment**: Financial market simulation and analysis with offline trading
- **Playwright Environment**: Web automation and interaction with browser control
- **FAISS Environment**: Vector search and similarity operations
- **Operator Browser Environment**: Advanced browser operations and automation

## 🧠 Model Management System

### ModelManager

The `ModelManager` provides centralized management of all language models with local proxy support:

```python
from src.infrastructures.models import model_manager

# Initialize ModelManager
await model_manager.initialize(use_local_proxy=False)  # Use official APIs
# await model_manager.initialize(use_local_proxy=True)  # Use local proxy

# List all available models
models = model_manager.list()
print(f"Available models: {models}")

# Get a specific model
model = model_manager.get("gpt-4")

# Create new models with different configurations
openai_model = model_manager.create_openai_model(
    "gpt-4",
    model_name="gpt-4",
    temperature=0.5,
    max_tokens=1000
)

claude_model = model_manager.create_anthropic_model(
    "claude-3-sonnet",
    model_name="claude-3-sonnet-20240229",
    temperature=0.7
)

google_model = model_manager.create_google_model(
    "gemini-pro",
    model_name="gemini-pro",
    temperature=0.3
)
```

### Available Models

#### OpenAI Models
- `gpt-4`: GPT-4 model
- `gpt-3.5-turbo`: GPT-3.5 Turbo model
- `gpt-4-turbo`: GPT-4 Turbo model
- `gpt-4.1`: Enhanced GPT-4 model

#### Anthropic Models
- `claude-3-sonnet`: Claude 3 Sonnet model
- `claude-3-opus`: Claude 3 Opus model
- `claude-3-haiku`: Claude 3 Haiku model
- `claude-2`: Claude 2 model

#### Google Models
- `gemini-pro`: Google Gemini Pro model
- `gemini-pro-vision`: Google Gemini Pro Vision model

#### Browser-Specific Models
- `bs-gpt-4.1`: Browser-specific GPT-4 model for web automation
- `o3`: Advanced model for research tasks

### Using Models with Agents

```python
# Create agent with specific model
agent = ToolCallingAgent(
    workdir="./workdir/researcher",
    model_name="gpt-4",  # Use OpenAI model
    prompt_name="tool_calling"
)

# Create agent with Anthropic model
agent = ToolCallingAgent(
    workdir="./workdir/writer", 
    model_name="claude-3-sonnet",  # Use Anthropic model
    prompt_name="tool_calling"
)

# Get model information
print(f"Current model: {agent.model}")
print(f"Agent name: {agent.name}")
```

## Tool Management System (TCP)

### Tool Context Protocol (TCP)

The Tool Context Protocol provides comprehensive management of all tools organized into specialized categories:

```python
from src.tools.protocol import tcp

# Initialize TCP
await tcp.initialize()

# List all available tools
tools = tcp.list()
print(f"Available tools: {tools}")

# Get a specific tool
tool = tcp.get("web_searcher")

# Execute tools asynchronously
result = await tcp.execute("web_fetcher", "https://example.com")

# Execute multiple tools concurrently
tool_calls = [
    {"name": "web_searcher", "args": ["AI research"], "kwargs": {}},
    {"name": "web_fetcher", "args": ["https://example.com"], "kwargs": {}},
]
results = await tcp.execute_multiple(tool_calls)
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

## Environment Management System (ECP)

### Environment Context Protocol (ECP)

The Environment Context Protocol provides unified management of specialized environments:

```python
from src.environments.protocol import ecp

# Initialize ECP
await ecp.initialize(["github", "file_system", "database"])

# List all available environments
environments = ecp.list()
print(f"Available environments: {environments}")

# Get a specific environment
env = ecp.get("github")

# Execute environment operations
result = await ecp.execute("github", "clone_repository", {
    "repository_url": "https://github.com/user/repo.git",
    "local_path": "./local_repo"
})
```

### Using Tools with Agents

```python
# Create agent with specific tools
agent = acp.get("tool_calling")

# Tools are configured in the config file and loaded automatically
available_tools = agent.list_available_tools()
print(f"Available tools: {available_tools}")
```

### Deep Researcher Tool Example

```python
from src.tools.workflow_tools import DeepResearcherTool

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

## Agent Management System (ACP)

### Agent Context Protocol (ACP)

The Agent Context Protocol provides comprehensive management of all agents:

```python
from src.agents.protocol import acp

# Initialize ACP
await acp.initialize(["tool_calling", "simple_chat", "debate_manager"])

# List all available agents
agents = acp.list()
print(f"Available agents: {agents}")

# Get a specific agent
agent = acp.get("tool_calling")

# Invoke agent
result = await acp.ainvoke("tool_calling", {
    "task": "Search for the latest AI news and summarize it"
})
```

## Prompt Management System

### PromptManager

The `PromptManager` provides centralized management of all prompt templates:

```python
from src.agents.prompts.prompt_manager import PromptManager

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
agent.add_tool("browser_tool")
agent.add_tool("deep_researcher_tool")

# Get agent information
agent_info = agent.get_agent_info()
print(f"Agent tools: {agent_info['tools']}")
```

## Advanced Usage

### Advanced Tool Usage with Different Models

```python
from src.tools.workflow_tools import DeepResearcherTool
from src.agents import ToolCallingAgent

# Create specialized agents with different models
researcher_agent = ToolCallingAgent(
    name="researcher",
    model_name="gpt-4",
    tools=["web_searcher", "web_fetcher", "deep_researcher"]
)

writer_agent = ToolCallingAgent(
    name="writer",
    model_name="claude-3-sonnet",
    tools=["web_searcher", "web_fetcher"]
)

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
from src.tools.protocol import tcp

# Initialize TCP
await tcp.initialize()

# Execute multiple tools concurrently
tool_calls = [
    {"name": "web_searcher_tool", "args": ["AI research"], "kwargs": {}},
    {"name": "web_fetcher_tool", "args": ["https://example.com"], "kwargs": {}},
    {"name": "deep_researcher_tool", "args": ["Quantum computing"], "kwargs": {}},
]

results = await tcp.execute_multiple(tool_calls)
for i, result in enumerate(results):
    print(f"Tool {i+1} result: {result}")
```

### Dynamic Configuration Changes

```python
# Change model dynamically
agent.change_model("claude-3-opus")

# Add new tools dynamically
agent.add_tool("browser_tool")
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

results = await tcp.execute_multiple(tool_calls)
```

### Async Model Invocation
```python
# Invoke models asynchronously
model = model_manager.get("gpt-4")
response = await model.ainvoke("Hello, how are you?")

# Generate multiple responses
messages = [HumanMessage(content="Hello"), HumanMessage(content="How are you?")]
responses = await model.agenerate(messages)
```

## Agent Types

### SimpleChatAgent
A conversational agent for human interaction that supports:
- Model management via ModelManager
- Prompt templates via PromptManager
- Async conversation capabilities
- State management and memory
- Simple chat interface without tool usage

### ToolCallingAgent
A specialized agent for tool execution that supports:
- Model management via ModelManager
- Tool management via TCP protocol
- Async tool usage with proper error handling
- Structured tool calling and response processing
- Integration with all tool categories (default, environment, agent, MCP)
- Support for complex workflow tools like DeepResearcherTool
- ThinkOutputBuilder for structured output formatting

### FinAgent
A financial analysis agent specialized for trading tasks that supports:
- Financial data analysis and processing
- Trading strategy implementation
- Market simulation and backtesting
- Financial metrics and performance evaluation
- Integration with trading environments

### DebateManagerAgent
A multi-agent coordination agent that supports:
- Coordinating multiple agents in debate sessions
- Managing debate rounds and turn-taking
- Facilitating agent-to-agent communication
- Debate topic management and moderation
- Async debate session handling

## Tool Integration

### Async Custom Tools
```python
from src.tools.protocol import tcp

# Initialize TCP
await tcp.initialize()
available_tools = tcp.list()
# Includes: web_searcher_tool, web_fetcher_tool, browser_tool, 
#          deep_researcher_tool, and MCP tools
```

### Async MCP Tools
```python
from src.tools.protocol import tcp

# Initialize TCP
await tcp.initialize()

# MCP tools are managed through the TCP protocol
# Tools are automatically loaded from the mcp_tools directory
mcp_tools = tcp.list()
print(f"MCP tools available: {mcp_tools}")
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
```python
from src.tools.workflow_tools import BrowserTool

# Initialize browser tool
browser = BrowserTool(model_name="gpt-4")

# Perform browser automation
result = await browser.arun("Navigate to example.com and take a screenshot")
print(result)
```

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
    from .base import browser_tool, deep_researcher_tool

tag = "tool_calling_agent"
workdir = f"workdir/{tag}"
log_path = "agent.log"

# IMPORTANT: Set use_local_proxy based on your API setup
use_local_proxy = False  # Set to False for official OpenAI/Anthropic APIs
# use_local_proxy = True   # Set to True for local proxy services (localhost:8000)
version = "0.1.0"

# Agent configuration
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

# Environment names
env_names = ["file_system", "github", "database"]

# Tool names (auto-loaded from tools directory)
tool_names = []

# Agent names
agent_names = ["tool_calling"]
```

### Base Configuration

```python
# configs/base.py
from mmengine.config import read_base
with read_base():
    from .environments.trading_offline import dataset, environment, metric

# Tool-specific configurations
browser_tool = dict(
    model_name = "bs-gpt-4.1",  # Browser-specific model
)

deep_researcher_tool = dict(
    model_name = "o3",  # Research-specific model
)
```

## 🎮 Running Examples

### Quick Start Examples
```bash
# Simple chat agent
python examples/run_simple_chat_agent.py

# Tool-calling agent with full capabilities
python examples/run_tool_calling_agent.py

# Financial analysis agent
python examples/run_finagent.py
```

### Advanced Examples
```bash
# Multi-agent debate system
python examples/debate/debate_frontend.py
# Access at http://localhost:5000
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
4. Register with TCP protocol

### Adding New Prompt Templates
1. Create Python file in `src/agents/prompts/templates/`
2. Define `PROMPT_TEMPLATES` dictionary or individual template variables
3. Templates are automatically loaded by PromptManager

### Adding New Routing Strategies
1. Create async routing function that takes `(message, state)` and returns agent name
2. Set as routing function in your custom agent system

## 🧪 Testing

### Run Test Scripts
```bash
# Protocol tests
python tests/test_acp.py
python tests/test_tcp.py
python tests/test_ecp.py

# Agent tests
python tests/test_agent.py
python tests/test_finagent.py

# Tool tests
python tests/test_browser.py
python tests/test_file_system.py

# Environment tests
python tests/test_github_system.py
python tests/test_faiss.py

# Model and infrastructure tests
python tests/test_models.py
python tests/test_memory_system.py

# Transformation tests
python tests/test_transformation.py
```

### Test Configuration
Most test scripts support configuration options:
```bash
# Run with specific config
python tests/test_acp.py --config configs/simple_chat_agent.py

# Run with custom options
python tests/test_models.py --cfg-options use_local_proxy=False
```

## 🔧 Troubleshooting

### Common Issues

#### API Connection Problems
- **Error**: "Failed to connect to OpenAI API"
  - **Solution**: Check `use_local_proxy` setting in config files
  - For official APIs: `use_local_proxy = False`
  - For local proxy: `use_local_proxy = True` (ensure proxy server is running)

#### Model Initialization Failures
- **Error**: "Model initialization failed"
  - **Solution**: Verify API keys in environment variables
  - Check model availability in ModelManager
  - Ensure network connectivity

#### Protocol Initialization Issues
- **Error**: "TCP/ECP/ACP initialization failed"
  - **Solution**: Check protocol configuration in config files
  - Verify tool/environment/agent names are correct
  - Ensure all dependencies are installed

#### Tool Loading Issues
- **Error**: "Tool not found" or "Tool initialization failed"
  - **Solution**: Verify tool registration in appropriate tool sets
  - Check tool dependencies installation
  - Review tool configuration in base.py

#### Environment Setup Issues
- **Error**: "Environment initialization failed"
  - **Solution**: Check environment-specific dependencies
  - Verify environment configuration files
  - Ensure required services are running (e.g., database, browser)

#### Transformation Issues
- **Error**: "Transformation failed"
  - **Solution**: Check transformation configuration
  - Verify source and target entities exist
  - Review transformation logic in transformation module

### Getting Help
- Check the [Issues](https://github.com/your-repo/issues) page for known problems
- Review the test files for usage examples
- Examine the example scripts in the `examples/` directory

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**: Prefer async implementations for consistency
4. **Add tests**: Ensure your changes are properly tested
5. **Submit a pull request**: Include a clear description of your changes

### Development Guidelines
- Follow the existing code style and patterns
- Add comprehensive tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration
- Powered by [LangChain](https://github.com/langchain-ai/langchain) for LLM integration
- Uses [Playwright](https://playwright.dev/) for browser automation
- Integrates with [FAISS](https://github.com/facebookresearch/faiss) for vector search

---

**AgentOrchestra** - Empowering AI agents with comprehensive tools and environments for real-world applications.