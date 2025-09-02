# Interactive Agent

## 概述

`InteractiveAgent` 是一个专为 Cursor 风格交互设计的 AI Agent，提供实时状态更新、用户交互和错误处理能力。

## 主要特性

### 🎮 交互式执行
- **实时状态显示**：每个步骤都有清晰的状态更新
- **用户控制**：可以在执行过程中暂停、继续或修改任务
- **进度跟踪**：显示当前进度和预计完成时间

### 🛠️ 工具集成
- 支持所有注册的工具（bash、file、python_interpreter、browser 等）
- 智能工具选择和参数验证
- 工具执行结果实时反馈

### 🧠 智能记忆
- 完整的执行历史记录
- 上下文感知的决策制定
- 错误恢复和学习能力

### 📊 状态管理
- 文件系统状态跟踪
- Todo 列表管理
- 结果累积和展示

## 使用方法

### 基本用法

```python
from src.agents.interactive_agent import InteractiveAgent

# 创建 agent
agent = InteractiveAgent(
    name="my_interactive_agent",
    model_name="gpt-4o",
    tools=["bash", "file", "python_interpreter"],
    interactive_mode=True,
    auto_continue=False
)

# 运行任务
result = await agent.run("Create a Python script that calculates fibonacci numbers")
```

### 命令行运行

```bash
# 基本运行
python examples/run_interactive_agent.py

# 指定任务
python examples/run_interactive_agent.py --task "Analyze the codebase and create documentation"

# 非交互模式
python examples/run_interactive_agent.py --no-interactive

# 自动继续模式
python examples/run_interactive_agent.py --auto-continue
```

## 配置选项

### Agent 配置

```python
agent = dict(
    name="interactive_agent",
    type="InteractiveAgent",
    model_name="gpt-4o",
    prompt_name="interactive",
    tools=["bash", "file", "project", "python_interpreter", "browser"],
    max_iterations=50,
    interactive_mode=True,
    auto_continue=False,
    max_steps=100,
    review_steps=10
)
```

### 交互设置

```python
interactive = dict(
    # 显示设置
    show_progress_bars=True,
    show_emoji=True,
    show_timestamps=True,
    
    # 用户交互设置
    ask_before_continue=True,
    confirm_dangerous_actions=True,
    allow_task_modification=True,
    
    # 错误处理
    max_retries=3,
    show_error_suggestions=True,
    auto_recovery=True
)
```

## 交互模式

### 1. 完全交互模式
- 每个步骤后询问用户是否继续
- 允许用户修改任务或提供指导
- 错误时提供多种处理选项

### 2. 半自动模式
- 正常执行，遇到错误时暂停
- 用户可以选择重试、跳过或修改
- 适合需要监督的自动化任务

### 3. 自动模式
- 完全自动化执行
- 错误时自动重试
- 适合批处理任务

## 状态显示格式

```
============================================================
🎯 CURRENT STATUS
============================================================
💭 Thinking: Analyzing the current file structure...
🎯 Next Goal: Create a comprehensive documentation file
🔧 Actions to execute: 3
  1. read_file
  2. write_file
  3. done
============================================================
```

## 错误处理

当遇到错误时，InteractiveAgent 会：

1. **显示错误详情**：清晰的错误信息和上下文
2. **提供解决方案**：建议的修复步骤
3. **用户选择**：让用户决定如何处理
4. **自动恢复**：尝试自动修复（如果启用）

## 与 ToolCallingAgent 的区别

| 特性 | ToolCallingAgent | InteractiveAgent |
|------|------------------|------------------|
| 交互性 | 基础 | 完全交互式 |
| 状态显示 | 简单日志 | 丰富的状态显示 |
| 用户控制 | 无 | 完全控制 |
| 错误处理 | 基础 | 智能错误处理 |
| 进度跟踪 | 基础 | 详细进度跟踪 |
| 配置灵活性 | 中等 | 高度可配置 |

## 最佳实践

### 1. 任务设计
- 将复杂任务分解为小步骤
- 使用清晰的描述和具体目标
- 提供必要的上下文信息

### 2. 交互设置
- 开发时使用完全交互模式
- 生产环境使用半自动或自动模式
- 根据任务复杂度调整 max_iterations

### 3. 工具选择
- 选择最合适的工具组合
- 避免不必要的工具依赖
- 确保工具参数正确

### 4. 错误处理
- 启用错误建议和自动恢复
- 设置合理的重试次数
- 记录错误历史以便改进

## 扩展开发

### 添加新的交互功能

```python
class CustomInteractiveAgent(InteractiveAgent):
    async def _custom_interaction(self):
        # 实现自定义交互逻辑
        pass
    
    async def _display_custom_status(self):
        # 实现自定义状态显示
        pass
```

### 自定义 Prompt 模板

```python
# 在 src/prompts/templates/ 中添加新的模板
CUSTOM_SYSTEM_PROMPT = """Your custom system prompt here"""
CUSTOM_AGENT_MESSAGE_PROMPT = """Your custom message prompt here"""
```

## 故障排除

### 常见问题

1. **Agent 卡住**：检查 max_iterations 设置
2. **工具执行失败**：验证工具配置和权限
3. **内存不足**：调整 memory 配置
4. **响应缓慢**：检查模型配置和网络连接

### 调试技巧

- 启用详细日志记录
- 使用较小的 max_iterations 进行测试
- 检查工具执行结果
- 验证 prompt 模板格式

## 总结

InteractiveAgent 提供了强大的交互式 AI 助手能力，特别适合需要用户监督和指导的复杂任务。通过合理的配置和工具选择，可以构建出高效、可靠的 AI 工作流程。
