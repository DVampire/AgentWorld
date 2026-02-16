# AgentWorld Examples

This directory contains example scripts demonstrating AgentWorld's capabilities. All examples follow the same pattern:

1. Load environment variables from `.env`
2. Parse config file via `--config` (with optional `--cfg-options` overrides)
3. Initialize the system (config, logger, model, tools, environments, agents)
4. Run the agent or experiment

## Table of Contents

- [Basic Agents](#basic-agents)
- [Trading Agents](#trading-agents)
- [Mobile and Browser Agents](#mobile-and-browser-agents)
- [Benchmarks](#benchmarks)
- [Optimization Experiments](#optimization-experiments)
- [Agent Optimization (AgentWorld Optimizers)](#agent-optimization-agentworld-optimizers)
- [Multi-Agent Systems](#multi-agent-systems)
- [Data Pipelines](#data-pipelines)
- [Utilities](#utilities)

---

## Basic Agents

### Simple Chat Agent

A minimal chat agent with no tools or environments -- useful for verifying your installation.

```bash
python examples/run_simple_chat_agent.py
```

- **Config:** `configs/simple_chat_agent.py`
- **What it does:** Sends a simple message ("Hello, how are you?") to a chat-only agent
- **Requirements:** Any LLM provider API key

### Tool-Calling Agent

An agent that can use external tools (web search, bash, python interpreter, etc.) to solve tasks.

```bash
python examples/run_tool_calling_agent.py
```

- **Config:** `configs/tool_calling_agent.py`
- **What it does:** Initializes the full stack (model, prompt, memory, tools, environments, agents) and sends a task to the agent. The agent reasons step-by-step and calls tools as needed.
- **Requirements:** LLM provider API key

### Planning Agent

A higher-level agent that decomposes complex tasks into sub-tasks and delegates to other agents (used as tools).

```bash
python examples/run_planning_agent.py
```

- **Config:** `configs/planning_agent.py`
- **What it does:** Transforms environments and sub-agents into callable tools, then sends a market analysis task to the planning agent
- **Requirements:** LLM provider API key

---

## Trading Agents

### Interday Trading

A day-level stock trading simulation agent.

```bash
python examples/run_interday_trading.py
```

- **Config:** `configs/interday_trading.py`
- **What it does:** Runs a trading agent that makes daily buy/sell decisions on AAPL in a simulated environment
- **Requirements:** LLM provider API key

### Intraday Trading

A minute-level stock trading simulation agent.

```bash
python examples/run_intraday_trading.py
```

- **Config:** `configs/intraday_trading.py`
- **What it does:** Same as interday but operates at minute-level granularity
- **Requirements:** LLM provider API key

### Online Trading Agent

A live crypto trading agent using the Hyperliquid exchange.

```bash
python examples/run_online_trading_agent.py
```

- **Config:** `configs/online_trading_agent.py`
- **What it does:** Connects to Hyperliquid with real account credentials and trades BTC
- **Requirements:** LLM provider API key, `HYPERLIQUID_ACCOUNTS` environment variable with account credentials

### Offline Trading Agent

A backtesting crypto trading agent using historical Hyperliquid data.

```bash
python examples/run_offline_trading_agent.py
```

- **Config:** `configs/offline_trading_agent.py`
- **What it does:** Same as online trading but uses historical data for backtesting
- **Requirements:** LLM provider API key, `HYPERLIQUID_ACCOUNTS` environment variable

### Trading Strategy Agent

An agent that generates and backtests trading strategies.

```bash
python examples/run_trading_strategy_agent.py
```

- **Config:** `configs/trading_strategy_agent.py`
- **What it does:** Generates a practical trading strategy, then backtests it using the `QuickBacktestEnvironment`
- **Requirements:** LLM provider API key

### ESG Agent

An Environmental, Social, and Governance (ESG) evaluation agent that processes ESG benchmark tasks.

```bash
python examples/run_esg_agent.py
```

- **Config:** `configs/esg_agent.py`
- **What it does:** Loads ESG dataset from `datasets/ESG`, runs evaluation tasks concurrently with resume support
- **Requirements:** LLM provider API key, ESG dataset in `datasets/ESG`

---

## Mobile and Browser Agents

### Mobile Agent

Automates Android device interactions via ADB.

```bash
python examples/run_mobile_agent.py
```

- **Config:** `configs/mobile_agent.py`
- **What it does:** Controls a mobile device to open Notes, type a message, and save it
- **Requirements:** LLM provider API key, ADB-connected Android device

### Anthropic Mobile Agent

Uses Anthropic's computer-use capabilities for mobile automation.

```bash
python examples/run_anthropic_mobile_agent.py
```

- **Config:** `configs/anthropic_mobile_agent.py`
- **What it does:** Same as mobile agent but leverages Anthropic's computer-use API
- **Requirements:** Anthropic API key, ADB-connected Android device

### Browser Agent

A web browser automation agent.

```bash
python examples/run_operator_browser.py
```

- **Config:** `configs/operator_browser_agent.py`
- **What it does:** Searches for the latest Apple news on Google using browser automation
- **Requirements:** LLM provider API key, Playwright installed (`playwright install`)

---

## Benchmarks

### General Benchmark Runner

Runs model inference on supported benchmarks (AIME24, AIME25, GPQA, GSM8K, LeetCode).

```bash
# Run AIME25 benchmark
python examples/run_benchmark.py --benchmark aime25 --max-concurrency 4

# Run GPQA benchmark
python examples/run_benchmark.py --benchmark gpqa --max-concurrency 8
```

- **Config:** `configs/tool_calling_agent.py`
- **CLI args:**
  - `--benchmark` : benchmark name (default: `leetcode`)
  - `--max-concurrency` : number of concurrent inference tasks (default: 4)
- **What it does:** Iterates through benchmark tasks, calls the model with structured output (reasoning + answer), evaluates and saves results
- **Requirements:** LLM provider API key

### LeetCode Agent

LeetCode benchmark with pipeline mode -- inference and evaluation run in parallel batches.

```bash
python examples/run_leetcode_agent.py --model openrouter/gemini-3-flash-preview --language python3
```

- **Config:** `configs/tool_calling_agent.py`
- **CLI args:**
  - `--model` : model to use for code generation
  - `--language` : programming language (e.g., `python3`, `kotlin`, `java`)
  - `--batch-size` : evaluation batch size
- **What it does:** Generates code solutions using structured output, submits to LeetCode via browser automation, evaluates correctness
- **Requirements:** LLM provider API key, Playwright, LeetCode cookie

### LeetCode Reflection Agent

LeetCode with iterative self-reflection -- generates, evaluates, and improves code over multiple rounds.

```bash
python examples/run_leetcode_reflection_agent.py --model openrouter/gemini-3-flash-preview --language python3 --max-rounds 3
```

- **Config:** `configs/tool_calling_agent.py`
- **CLI args:**
  - `--model` : model to use
  - `--language` : programming language
  - `--max-rounds` : maximum reflection iterations (default: 3)
  - `--opt-threshold` : optimization threshold percentage (default: 65.0)
  - `--judge-mode` : `threshold` or `llm` for deciding when to optimize
- **What it does:** Generates code, evaluates, reflects on failures, generates improved code -- repeating up to `max-rounds` times
- **Requirements:** LLM provider API key, Playwright, LeetCode cookie

### LeetCode Tool Demo

A standalone demo of the LeetCode problem-fetching tool (no agent needed).

```bash
python examples/run_leetcode_tool.py
```

- **Config:** None
- **What it does:** Fetches LeetCode problem information by slug or ID (e.g., "two-sum", "reverse-linked-list")
- **Requirements:** `.env` file

---

## Optimization Experiments

These scripts compare direct model inference against optimization-enhanced approaches on academic benchmarks. They use standalone TextGrad/GRPO/REINFORCE++ implementations.

### AIME Experiments

Compare optimization methods on AIME math competition problems (AIME24, AIME25).

```bash
# TextGrad optimization
python examples/run_aime_textgrad_experiment.py --models gpt-4o --max-step 3

# Reflection loops (no gradient)
python examples/run_aime_reflection_experiment.py --models gpt-4.1 --max-step 3

# GRPO or REINFORCE++ optimization
python examples/run_aime_optimizer_experiment.py --models gpt-4o --optimizer reinforce++ --num-steps 1
```

- **CLI args (common):**
  - `--models` : models to evaluate
  - `--datasets` : datasets to use (default: AIME24, AIME25)
  - `--max-step` / `--num-steps` : optimization iterations
  - `--max-samples` : limit number of samples
  - `--use-tool-agent` : use tool-calling agent instead of direct model inference
  - `--output` : output directory
- **What they do:** Run each sample through direct inference and optimized inference, then compare accuracy
- **Requirements:** LLM provider API key

### GPQA / GSM8K Experiments

Same optimization methods applied to GPQA (science) and GSM8K (math) benchmarks.

```bash
# TextGrad optimization
python examples/run_gpqa_gsm8k_textgrad_experiment.py --models gpt-4o

# Reflection loops
python examples/run_gpqa_gsm8k_reflection_experiment.py --models gpt-4.1

# GRPO or REINFORCE++
python examples/run_gpqa_gsm8k_optimizer_experiment.py --optimizer grpo --use-tool-agent
```

- **CLI args:** Same structure as AIME experiments, with `--datasets` defaulting to GPQA_diamond and GSM8K_test
- **Requirements:** LLM provider API key

---

## Agent Optimization (AgentWorld Optimizers)

These scripts use AgentWorld's built-in optimizer framework (`src.optimizer`) to optimize tool-calling agents on benchmarks.

### Sequential Experiment

Runs optimizer experiments sequentially on benchmark tasks.

```bash
python examples/run_tool_calling_agent_experiment.py --optimizer reflection --benchmark gpqa
```

- **Config:** `configs/tool_calling_agent.py`
- **CLI args:**
  - `--optimizer` : `grpo`, `reinforce_pp`, or `reflection`
  - `--benchmark` : benchmark name (e.g., `gpqa`, `aime25`, `leetcode`)
- **Requirements:** LLM provider API key

### Async Experiment (Concurrent)

Concurrent version with resume support, configurable batch sizes, and detailed result tracking.

```bash
python examples/run_tool_calling_agent_experiment_async.py \
  --optimizer reflection \
  --benchmark gpqa \
  --concurrency 8 \
  --split test
```

- **Config:** `configs/tool_calling_agent.py`
- **CLI args:**
  - `--optimizer` : `grpo`, `reinforce_pp`, or `reflection`
  - `--benchmark` : benchmark name
  - `--concurrency` : number of concurrent tasks (default: 16)
  - `--split` : dataset split (`train` or `test`)
  - `--batchsize` : batch size for processing
  - `--model_name` : override model
  - `--resume` : resume from previous results
- **Requirements:** LLM provider API key

### TextGrad Agent Optimization

Optimizes agent prompts using TextGrad.

```bash
python examples/run_tool_calling_agent_optimized.py
```

- **Config:** `configs/tool_calling_agent.py`
- **What it does:** Runs 3 TextGrad optimization steps on the agent's prompts, then runs the agent with optimized prompts
- **Requirements:** LLM provider API key (uses `gpt-4o` for optimization)

### Reflection Agent Optimization

Optimizes agent prompts using the Reflection optimizer.

```bash
python examples/run_tool_calling_agent_reflection.py
```

- **Config:** `configs/tool_calling_agent.py`
- **What it does:** Creates a `ReflectionOptimizer` that iteratively improves both trainable variables and solution approach
- **Requirements:** LLM provider API key

---

## Multi-Agent Systems

### Debate Frontend

A Flask web application for real-time multi-agent debate visualization.

```bash
python examples/debate/debate_frontend.py
```

- **Config:** `configs/multi_agent_debate.py`
- **What it does:** Starts a web server at `http://0.0.0.0:8000` with a UI for watching two agents debate topics in real time via WebSocket streaming
- **Requirements:** LLM provider API key

---

## Data Pipelines

### Download Market Data

Download crypto market data from Binance.

```bash
# Using the shell script
bash examples/run_download.sh

# Or directly
python examples/download/download.py --config configs/download/crypto/crypto_binance_price_1day.py
python examples/download/download.py --config configs/download/crypto/crypto_binance_price_1min.py
```

- **Config:** `configs/download/crypto/crypto_binance_price_*.py`
- **What it does:** Downloads OHLCV price data from Binance at the specified interval
- **Requirements:** Binance API key (if applicable)

### Process Market Data

Process downloaded market data for use by trading agents.

```bash
# Using the shell script
bash examples/run_process.sh

# Or directly
python examples/process/process.py --config configs/process/crypto.py
```

- **Config:** `configs/process/crypto.py`
- **What it does:** Transforms raw market data into the format expected by trading environments

---

## Utilities

### Analysis (Re-run Incorrect Tasks)

Re-runs only the incorrect tasks from a previous experiment.

```bash
python examples/analysis.py \
  --optimizer reflection \
  --benchmark gpqa \
  --experiment_file path/to/results.json \
  --concurrency 4
```

- **Config:** `configs/tool_calling_agent.py`
- **What it does:** Loads previous experiment results, identifies incorrect answers, and re-runs only those tasks with the optimizer

### Extract Incorrect Samples

Extracts incorrect samples from experiment results into a separate file.

```bash
python examples/extract_incorrect_samples.py
```

- **What it does:** Reads a results JSON, filters for incorrect answers, outputs a new file with statistics
- **Note:** Paths are hardcoded in the script; edit before running

---

## Common CLI Options

All example scripts support these arguments:

| Argument | Description |
|---|---|
| `--config` | Path to the config file (default varies per script) |
| `--cfg-options` | Inline config overrides (e.g., `model_name=openrouter/gpt-4o workdir=workdir/test`) |

### Examples of Config Overrides

```bash
# Change model
python examples/run_tool_calling_agent.py --cfg-options model_name=openrouter/gpt-4o

# Change working directory
python examples/run_tool_calling_agent.py --cfg-options workdir=workdir/my_experiment

# Change max tokens
python examples/run_tool_calling_agent.py --cfg-options max_tokens=8192
```
