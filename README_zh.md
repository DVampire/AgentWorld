# AgentWorld（中文）

[English](README.md) | 中文

AgentWorld 是一个 **自进化（self-evolving）智能体框架**。它提供模块化的运行时与组件体系，用于构建能够通过结构化反馈闭环不断 **迭代改进** 的智能体（例如反思式优化、基于奖励信号的优化等），同时保持系统在工程上可组合、可观测、可演化。

## 自进化闭环概览

从机制上看，AgentWorld 支持反复迭代的闭环：

- **行动（Act）**：智能体基于 LLM + 可用工具/环境产生行动与输出。
- **观测（Observe）**：记录结果、轨迹、关键中间信息，以及来自环境的反馈信号。
- **优化（Optimize）**：用优化器把反馈转成“可复用的改进”（提示词/解法/变量等的更新），例如 Reflection、GRPO、Reinforce++。
- **记忆（Remember）**：把会话事件、总结与洞察写入记忆系统，供后续步骤/会话复用。

## 核心模块

- **Agents（`src/agent/`）**：决定“下一步做什么”的运行逻辑（规划、工具调用、领域 agent 等）。
- **Tools（`src/tool/`）**：暴露给智能体调用的能力（工作流工具 + 默认工具）。
- **Environments（`src/environment/`）**：可交互的有状态接口（文件系统、回测环境、浏览器/移动端环境等）。
- **Memory（`src/memory/`）**：会话/事件记忆系统，用于总结、洞察与长期状态。
- **Optimizers（`src/optimizer/`）**：自进化算法，把反馈转为可落地的更新（reflection、GRPO、Reinforce++ 等）。
- **Tracing & Versioning（`src/tracer/`, `src/version/`）**：记录轨迹并管理迭代产物，便于分析与复现实验。
- **配置系统（`configs/`, `src/config/`）**：MMEngine 风格的组合式配置，用于一致地装配 agent/tool/env/memory/model。

## 设计目标

- **可组合**：能在不推翻整体架构的情况下替换/扩展 agent、工具、环境、记忆、优化器。
- **可观测**：结构化的 trace 与 memory event 让失败分析与改进步骤更透明。
- **可演化**：通过优化器 + 持久化记忆实现迭代式改进，而不是一次性推理。

## 目录结构

```
AgentWorld/
  configs/                 # 组件组合配置（agent/tool/env/memory/model）
  src/
    agent/                 # agents
    environment/           # environments
    tool/                  # tools
    memory/                # memory systems
    optimizer/             # 自进化优化器
    model/                 # model manager + provider backends
    prompt/                # prompt templates / prompt manager
    tracer/                # tracing
    version/               # versioning
  libs/                    # 内置/引入的库
  workdir/                 # 运行产物（日志/轨迹/结果等）
```

## （可选）运行 Tool-Calling Agent

前置条件：
- 先在你的环境里安装好依赖
- 将 `.env.template` 复制为 `.env`，并配置一个模型提供方的 key（例如 `OPENROUTER_API_KEY=...`）

示例：

```bash
python examples/run_tool_calling_agent.py --config configs/tool_calling_agent.py
```

覆盖模型/输出目录：

```bash
python examples/run_tool_calling_agent.py \
  --config configs/tool_calling_agent.py \
  --cfg-options model_name=openrouter/gpt-4o workdir=workdir/demo tag=demo
```

