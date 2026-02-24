# AgentWorld（中文）

[English](README.md) | 中文

AgentWorld 是一个 Python 智能体框架：通过统一的配置系统，把 **模型（Models）**、**提示词（Prompts）**、**记忆（Memory）**、**工具（Tools）**、**环境（Environments）**、**评测（Benchmarks）** 串起来，支持一键运行与批量实验。

## 你可以做什么

- **Tool-Calling Agent**：按配置启动通用智能体，自动规划并调用工具/环境完成任务。
- **Benchmarks**：在内置数据集上跑 AIME24/25、GSM8K、GPQA（LeetCode 为进阶可选）。
- **优化器实验**：在整套 benchmark 上并发跑 Reflection / GRPO / Reinforce++，并自动续跑与保存结果。

## 环境要求

- **Python**：推荐 **Python 3.11**（仓库内包含子模块对 >=3.10 / >=3.11 有要求）。
- **系统**：macOS/Linux/Windows 均可；部分流程（如 LeetCode）依赖 Playwright + 浏览器。

## 安装

创建虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
```

> 说明  
> `requirements.txt` 覆盖面较大（例如包含 `torch`），安装会比较重；如果你只跑部分功能，后续可以再按需精简依赖。

## 环境变量

多数入口脚本都会 `load_dotenv()`，因此你可以把 key 写在根目录 `.env`。

### 模型提供方

任选一种提供方（推荐 OpenRouter），设置对应 API key：

```bash
# OpenRouter（推荐）
OPENROUTER_API_KEY=...
# 可选（OpenRouter headers）
OPENROUTER_API_BASE=        # optional; defaults to OpenRouter API base
OPENROUTER_HTTP_REFERER=
OPENROUTER_X_TITLE=

# OpenAI（可选）
OPENAI_API_KEY=...
OPENAI_API_BASE=

# Anthropic（可选）
ANTHROPIC_API_KEY=...
ANTHROPIC_API_BASE=

# Google Gemini（可选）
GOOGLE_API_KEY=...
```

## 快速开始：Tool-Calling Agent

推荐入口：
- `examples/run_tool_calling_agent.py`
- 默认配置：`configs/tool_calling_agent.py`

运行：

```bash
python examples/run_tool_calling_agent.py --config configs/tool_calling_agent.py
```

### 命令行覆盖配置

所有脚本都支持 `--cfg-options`（MMEngine 风格）覆盖配置键值。

示例：切换模型 + 输出目录（建议更换 workdir，避免覆盖历史输出）：

```bash
python examples/run_tool_calling_agent.py \
  --config configs/tool_calling_agent.py \
  --cfg-options model_name=openrouter/gpt-4o workdir=workdir/my_run tag=my_run
```

输出（日志、benchmark 产物等）默认在 `workdir/<tag>/` 下。

## 基准评测（Benchmarks）

### 内置 benchmark 名称

- `aime24`（数据集：`datasets/AIME24`）
- `aime25`（数据集：`datasets/AIME25`）
- `gsm8k`（数据集：`datasets/gsm8k`）
- `gpqa`（数据集：`datasets/GPQA`）
- `leetcode`（数据集：`datasets/leetcode`，进阶流程）

### 跑整套数据集的实验（推荐）

推荐用 `examples/run_tool_calling_agent_experiment_async.py`：

- 支持 `--benchmark`, `--optimizer`, `--concurrency`, `--split`, `--model_name`
- 结果写入 `workdir/.../results/*.json`
- 默认会尝试从最近一次结果文件 **resume**；想从头跑建议换一个 `workdir/tag`（或删除旧 results）

示例（Reflection，并发 8）：

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

可选优化器：
- `reflection`
- `grpo`
- `reinforce_pp`

### 单脚本跑 AIME（示例）

`examples/run_benchmark.py` 用真实模型做完整循环评测，结果会保存到 `examples/workdir/results/`（脚本内固定路径）。

```bash
python examples/run_benchmark.py \
  --config configs/tool_calling_agent.py \
  --benchmark aime25 \
  --max-concurrency 4
```

## （可选）LeetCode benchmark 前置条件

LeetCode benchmark 会用 Playwright 控制 GitHub Codespace（浏览器里的 VS Code），并通过 LeetCode 插件提交评测。

需要的环境变量（仅变量名；不要提交密钥）：

```bash
GITHUB_USERNAME=
GITHUB_PASSWORD=
GITHUB_PROJECT_URL=
GITHUB_CODESPACE_URL=

LEETCODE_USERNAME=
LEETCODE_PASSWORD=
LEETCODE_COOKIE=           # 可选：可加速登录
```

## 项目结构

```
AgentWorld/
  configs/                 # MMEngine 风格配置文件
  examples/                # 可运行入口脚本
  src/
    agent/                 # agents（tool_calling / planning / trading / ...）
    benchmark/             # benchmark 实现 + manager
    environment/           # environments（filesystem / quickbacktest / browser / ...）
    memory/                # memory systems
    model/                 # model manager + provider backends
    tool/                  # tools（default_tools + workflow_tools）
  datasets/                # benchmark datasets（AIME / GSM8K / GPQA / LeetCode / ...）
  libs/                    # 内置/引入的库（browser-use / EvoAgentX / textgrad / ...）
  workdir/                 # 运行输出（logs / results / artifacts）
```

## 常见问题

- **缺少 API key**：在 `.env` 设置 `OPENROUTER_API_KEY`（或其他提供方的 key）。
- **想从头跑**：换一个 `workdir/tag`（`--cfg-options`）避免结果混在一起。
- **依赖很重**：`requirements.txt` 包含很多包；建议用干净环境安装，后续按需精简。

