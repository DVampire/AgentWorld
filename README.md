# AgentWorld

English | [中文说明](README_zh.md)

AgentWorld is a **self-evolving agent framework** in Python. It provides a modular runtime for building agents that can **learn / improve over time** via structured feedback loops (e.g., reflection, reward-driven optimization), while keeping the system decomposed into explicit building blocks.

## Self-evolution at a glance

At a high level, AgentWorld supports an iterative loop:

- **Act**: an agent produces actions/outputs using an LLM and the available tools/environments.
- **Observe**: capture outcomes, traces, intermediate reasoning, and environment feedback.
- **Optimize**: update prompts/solutions/variables using an optimizer (e.g., reflection or RL-style methods).
- **Remember**: persist summaries/insights/records to memory for later steps and sessions.

## Core building blocks

- **Agents (`src/agent/`)**: runtime logic that decides *what to do next* (planning, tool-calling, domain agents, etc.).
- **Tools (`src/tool/`)**: callable capabilities exposed to agents (workflow tools + default tools).
- **Environments (`src/environment/`)**: stateful interfaces that tools/agents can interact with (filesystem, trading backtest envs, browser/mobile envs, etc.).
- **Memory (`src/memory/`)**: session/event memory systems for summarization, insights, and long-term state.
- **Optimizers (`src/optimizer/`)**: self-improvement algorithms that turn feedback into updated prompts/solutions/variables (reflection, GRPO, Reinforce++, etc.).
- **Tracing & versioning (`src/tracer/`, `src/version/`)**: record trajectories and manage iterative artifacts across runs.
- **Config system (`configs/`, `src/config/`)**: MMEngine-style configs to compose agents/tools/envs/memory/models consistently.

## Design goals

- **Composable**: add/replace agents, tools, environments, memory systems, and optimizers without rewriting the whole stack.
- **Inspectable**: structured traces and memory events make it easier to analyze failures and improvement steps.
- **Evolvable**: explicit optimizers + persistent memory enable iterative refinement rather than one-shot inference.

## Repository layout

```
AgentWorld/
  configs/                 # config composition (agents/tools/envs/memory/models)
  src/
    agent/                 # agents
    environment/           # environments
    tool/                  # tools
    memory/                # memory systems
    optimizer/             # self-evolution optimizers
    model/                 # model manager + provider backends
    prompt/                # prompt templates / prompt manager
    tracer/                # tracing
    version/               # versioning
  libs/                    # vendored libraries
  workdir/                 # runtime artifacts (logs, traces, results, etc.)
```