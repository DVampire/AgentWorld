# AgentWorld

A modular multi-agent framework for building, running, optimizing, and evaluating LLM-powered agents. AgentWorld provides a protocol-based architecture where agents interact with environments and tools to accomplish tasks, with built-in support for prompt optimization via TextGrad, Reflection, GRPO, and REINFORCE++.

## Architecture

AgentWorld is organized around three core context protocols:

| Protocol | Description | Singleton |
|---|---|---|
| **ACP** (Agent Context Protocol) | Manages agent registration, lifecycle, and invocation | `acp` |
| **ECP** (Environment Context Protocol) | Manages environments that provide state and actions | `ecp` |
| **TCP** (Tool Context Protocol) | Manages tools that agents can call | `tcp` |

Supporting systems include the **ModelManager** (multi-provider LLM routing), **PromptManager** (hierarchical trainable templates), **MemoryManager** (event-based long-term memory), **BenchmarkManager** (evaluation harness), and **VersionManager** (component versioning).

```
Agent  ──calls──>  Tool (via TCP)
  │                  │
  ├──observes──>  Environment (via ECP)
  │                  │
  ├──remembers──> Memory
  │
  └──thinks via──> Model (OpenAI / Anthropic / Google / OpenRouter)
```

## Project Structure

```
AgentWorld/
├── configs/                  # Configuration files (mmengine Python configs)
│   ├── base.py               # Base config with shared defaults
│   ├── agents/               # Per-agent configs
│   ├── environments/         # Per-environment configs
│   ├── tools/                # Per-tool configs
│   ├── memory/               # Memory system configs
│   ├── download/             # Data download configs
│   └── process/              # Data processing configs
├── src/                      # Source code
│   ├── agent/                # Agent base class + 13 concrete agents
│   ├── environment/          # Environment base class + 14 environments
│   ├── tool/                 # Tool base class + 12 tools
│   ├── model/                # Multi-provider LLM manager
│   ├── memory/               # Event-based memory with insights/summaries
│   ├── optimizer/            # TextGrad, Reflection, GRPO, REINFORCE++
│   ├── prompt/               # Hierarchical prompt templates (Variable system)
│   ├── benchmark/            # AIME, GPQA, GSM8K, LeetCode benchmarks
│   ├── config/               # Singleton config loader
│   ├── transformation/       # Cross-protocol adapters (Agent<->Tool<->Env)
│   ├── dynamic/              # Runtime code generation and class loading
│   ├── session/              # Session context for concurrent execution
│   ├── version/              # Component version tracking
│   ├── tracer/               # Execution trace recording
│   ├── message/              # Message types (System, Human, AI, Tool)
│   ├── registry.py           # mmengine registries for auto-discovery
│   └── utils/                # Utilities (path, token, string, etc.)
├── examples/                 # Example scripts (see examples/README.md)
├── datasets/                 # Benchmark datasets (AIME, GPQA, GSM8K, LeetCode, ESG)
├── tests/                    # Test suite
├── docs/                     # Documentation assets
├── libs/                     # External library references
├── requirements.txt          # Python dependencies
└── environment.yml           # Conda environment specification
```

## Prerequisites

- **Python** 3.11+
- **Conda** (recommended for environment management)
- **Git**
- API keys for at least one LLM provider:
  - [OpenAI](https://platform.openai.com/) (for GPT-4o, GPT-4.1, GPT-5, o3, etc.)
  - [Anthropic](https://console.anthropic.com/) (for Claude Sonnet/Opus)
  - [Google AI](https://ai.google.dev/) (for Gemini)
  - [OpenRouter](https://openrouter.ai/) (unified access to multiple providers)
- **Optional** (for specific use cases):
  - [Playwright](https://playwright.dev/) for browser automation agents
  - ADB (Android Debug Bridge) for mobile agents
  - [Alpaca](https://alpaca.markets/) or [Binance](https://www.binance.com/) accounts for trading agents
  - [Hyperliquid](https://hyperliquid.xyz/) account for crypto trading agents

## Installation

### Option 1: Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/DVampire/AgentWorld.git
cd AgentWorld

# Create conda environment from the specification
conda env create -f environment.yml

# Activate the environment
conda activate agentworld
```

### Option 2: pip

```bash
# Clone the repository
git clone https://github.com/DVampire/AgentWorld.git
cd AgentWorld

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: Playwright Setup

If you plan to use browser-based agents or the LeetCode benchmark:

```bash
playwright install
```

### Optional: TA-Lib Setup

For trading agents that use technical analysis indicators, TA-Lib requires a system-level library:

```bash
# Ubuntu/Debian
sudo apt-get install libta-lib-dev

# macOS
brew install ta-lib

# Then install the Python wrapper (included in requirements.txt)
pip install TA-Lib
```

## Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
# Required: At least one LLM provider
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...

# Optional: Trading services
ALPACA_API_KEY=...
ALPACA_SECRET_KEY=...
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...

# Optional: Hyperliquid (JSON string with account credentials)
HYPERLIQUID_ACCOUNTS='[{"account_address": "...", "secret_key": "...", "vault_address": "..."}]'

# Optional: GitHub integration
GITHUB_TOKEN=ghp_...

# Optional: LeetCode (for benchmark submission)
LEETCODE_COOKIE=...
```

## Quick Start

### 1. Simple Chat Agent (Minimal Setup)

The simplest way to verify your installation:

```bash
python examples/run_simple_chat_agent.py
```

This runs a basic chat agent with no tools or environments -- just an LLM conversation.

### 2. Tool-Calling Agent

An agent that can use tools (web search, bash, python interpreter, etc.) to solve tasks:

```bash
python examples/run_tool_calling_agent.py
```

### 3. Planning Agent

A higher-level agent that decomposes complex tasks and delegates to sub-agents:

```bash
python examples/run_planning_agent.py
```

### 4. Run a Benchmark

Evaluate model performance on math/science/coding benchmarks:

```bash
python examples/run_benchmark.py --benchmark aime25 --max-concurrency 4
```

See [examples/README.md](examples/README.md) for the full list of examples and use cases.

## Configuration

AgentWorld uses [mmengine](https://github.com/open-mmlab/mmengine)-style Python config files. Configs are hierarchical -- scenario configs inherit from `configs/base.py` and compose component configs.

### Overriding Config Values

All example scripts support `--cfg-options` for inline overrides:

```bash
# Change the model
python examples/run_tool_calling_agent.py --cfg-options model_name=openrouter/gpt-4o

# Change working directory
python examples/run_tool_calling_agent.py --cfg-options workdir=workdir/my_experiment
```

### Config Structure

A typical scenario config (e.g., `configs/tool_calling_agent.py`):

```python
from configs.base import *                   # Inherit base defaults
from configs.agents.tool_calling import *    # Agent config
from configs.tools.browser import *          # Tool configs
from configs.environments.file_system import * # Environment configs
from configs.memory.general_memory_system import * # Memory config

# Override base values
tag = "tool_calling"
workdir = f"workdir/{tag}"

# Declare which components to activate
agent_names = ["tool_calling"]
tool_names = ["browser", "bash", "python_interpreter"]
env_names = ["file_system"]
memory_names = ["general_memory_system"]
```

## Key Concepts

### Agents
Agents are the primary actors. They observe environment state, reason via LLMs, and call tools. Built-in agents include: `ToolCallingAgent`, `PlanningAgent`, `SimpleChatAgent`, `TradingStrategyAgent`, `InterdayTradingAgent`, `IntradayTradingAgent`, `OnlineTradingAgent`, `OfflineTradingAgent`, `ESGAgent`, `MobileAgent`, `OperatorBrowserAgent`, and more.

### Environments
Environments provide observable state and expose actions. Actions are registered via the `@ecp.action()` decorator. Built-in environments include: `FileSystemEnvironment`, `GitHubEnvironment`, `DatabaseEnvironment`, `FaissEnvironment`, `AlpacaEnvironment`, `BinanceEnvironment`, `InterdayTradingEnvironment`, `QuickBacktestEnvironment`, `MobileEnvironment`, `OperatorBrowserEnvironment`, and more.

### Tools
Tools are callable functions that agents invoke. Built-in tools include: `BashTool`, `PythonInterpreterTool`, `WebSearcherTool`, `WebFetcherTool`, `BrowserTool`, `PlotterTool`, `DeepResearcherTool`, `RetrieverTool`, `MdifyTool`, `DoneTool`, and more.

### Optimizers
AgentWorld supports four prompt optimization strategies that treat prompt text as differentiable variables:
- **TextGrad**: Computes "text gradients" (natural language feedback) and applies gradient descent to prompts
- **Reflection**: LLM-based self-reflection that evaluates and improves prompt variables iteratively
- **GRPO**: Group Relative Policy Optimization -- generates candidate prompts and selects by reward
- **REINFORCE++**: Reinforcement learning approach to prompt optimization

### Transformations
Components can be adapted across protocols: an environment can be wrapped as a tool (`E2T`), an agent can be wrapped as a tool (`A2T`), and vice versa. This enables compositional architectures like the `PlanningAgent`, which uses other agents as callable tools.

## License

See [LICENSE](LICENSE) for details.
