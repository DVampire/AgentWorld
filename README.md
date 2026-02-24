# AgentWorld

English | [中文说明](README_zh.md)

AgentWorld is a Python agent framework that wires together **Models**, **Prompts**, **Memory**, **Tools**, **Environments**, and **Benchmarks** with a unified config system.

## What you can do

- **Tool-Calling Agent**: run a general-purpose agent that plans and calls tools/envs based on config.  
- **Benchmarks**: evaluate on built-in datasets like AIME24/25, GSM8K, GPQA (and LeetCode as an advanced option).  
- **Optimizer experiments**: run Reflection / GRPO / Reinforce++ over full benchmarks with concurrency and auto-resume results.  

## Requirements

- **Python**: recommended **Python 3.11** (repo includes submodules that require >=3.10 / >=3.11).  
- **OS**: macOS/Linux/Windows should work; some benchmark flows (e.g. LeetCode) require Playwright + a browser.  

## Install

Create a virtualenv and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
```

> Note  
> `requirements.txt` is intentionally broad and may be heavy (e.g. `torch`). If you only need a subset, you can later slim it down.

## Environment variables

Most entry scripts call `load_dotenv()`, so you can put keys in a root `.env` file.

### Model providers

Pick **one** provider (OpenRouter recommended) and set the corresponding API key.

```bash
# OpenRouter (recommended)
OPENROUTER_API_KEY=...
# Optional (OpenRouter headers)
OPENROUTER_API_BASE=        # optional; defaults to OpenRouter API base
OPENROUTER_HTTP_REFERER=
OPENROUTER_X_TITLE=

# OpenAI (optional)
OPENAI_API_KEY=...
OPENAI_API_BASE=

# Anthropic (optional)
ANTHROPIC_API_KEY=...
ANTHROPIC_API_BASE=

# Google Gemini (optional)
GOOGLE_API_KEY=...
```

## Quickstart: Tool-Calling Agent

The canonical entry is:

- `examples/run_tool_calling_agent.py`
- default config: `configs/tool_calling_agent.py`

Run it:

```bash
python examples/run_tool_calling_agent.py --config configs/tool_calling_agent.py
```

### Override config from CLI

All scripts support `--cfg-options` (MMEngine style) to override config keys.

Example: change model + workdir (recommended to avoid clobbering prior outputs).

```bash
python examples/run_tool_calling_agent.py \
  --config configs/tool_calling_agent.py \
  --cfg-options model_name=openrouter/gpt-4o workdir=workdir/my_run tag=my_run
```

Outputs (logs, benchmark artifacts, etc.) go under `workdir/<tag>/`.

## Benchmarks

### Built-in benchmark names

- `aime24` (dataset path: `datasets/AIME24`)  
- `aime25` (dataset path: `datasets/AIME25`)  
- `gsm8k` (dataset path: `datasets/gsm8k`)  
- `gpqa` (dataset path: `datasets/GPQA`)  
- `leetcode` (dataset path: `datasets/leetcode`, advanced flow)

### Dataset-wide experiments (recommended)

Use `examples/run_tool_calling_agent_experiment_async.py`:

- supports `--benchmark`, `--optimizer`, `--concurrency`, `--split`, `--model_name`
- writes results to `workdir/.../results/*.json`
- by default it will try to **resume** from the latest results file; to start fresh, use a new `workdir`/`tag` (or delete the old results).

Example (Reflection, concurrency 8):

```bash
python examples/run_tool_calling_agent_experiment_async.py \
  --config configs/tool_calling_agent.py \
  --benchmark gpqa \
  --optimizer reflection \
  --concurrency 8 \
  --split test \
  --model_name openrouter/claude-sonnet-4.5 \
  --cfg-options workdir=workdir/gpqa_reflection tag=gpqa_reflection
```

Optimizers:
- `reflection`
- `grpo`
- `reinforce_pp`

### Single benchmark loop (AIME example)

`examples/run_benchmark.py` runs a full-loop benchmark with real model inference and saves results under `examples/workdir/results/`.

```bash
python examples/run_benchmark.py \
  --config configs/tool_calling_agent.py \
  --benchmark aime25 \
  --max-concurrency 4
```

## (Optional) LeetCode benchmark prerequisites

LeetCode benchmark uses Playwright to control a GitHub Codespace (VS Code in browser) and submit solutions via LeetCode extension.

Required env vars (names only; do NOT commit secrets):

```bash
GITHUB_USERNAME=
GITHUB_PASSWORD=
GITHUB_PROJECT_URL=
GITHUB_CODESPACE_URL=

LEETCODE_USERNAME=
LEETCODE_PASSWORD=
LEETCODE_COOKIE=           # optional; can speed up login
```

## Project structure

```
AgentWorld/
  configs/                 # MMEngine-style config files
  examples/                # runnable entry scripts
  src/
    agent/                 # agents (tool_calling, planning, trading, ...)
    benchmark/             # benchmark implementations + manager
    environment/           # environments (filesystem, quickbacktest, browser, ...)
    memory/                # memory systems
    model/                 # model manager + provider backends
    tool/                  # tools (default_tools + workflow_tools)
  datasets/                # benchmark datasets (AIME, GSM8K, GPQA, LeetCode, ...)
  libs/                    # vendored libraries (browser-use, EvoAgentX, textgrad, ...)
  workdir/                 # runtime outputs (logs, results, artifacts)
```

## Troubleshooting

- **Missing API key**: set `OPENROUTER_API_KEY` (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`) in `.env`.  
- **Want a clean run**: use a new `--cfg-options workdir=... tag=...` to avoid mixing outputs.  
- **Heavy dependencies**: `requirements.txt` is large; consider using a clean env and installing only what you need later.  