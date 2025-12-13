# TextGrad 优化器

使用 TextGrad 框架自动优化 Agent 提示词的模块。通过迭代执行 Agent 任务并根据执行结果自动调整提示词，提升 Agent 性能。

## 目录


- [快速开始](#快速开始)
- [使用方式](#使用方式)
  - [方式1：便捷函数（推荐）](#方式1便捷函数推荐)
  - [方式2：使用类（更灵活）](#方式2使用类更灵活)
- [参数说明](#参数说明)
- [工作原理](#工作原理)
- [注意事项](#注意事项)




## 快速开始

```python
from src.optimizers.textgrad_optimizer import optimize_agent_with_textgrad

# 获取 Agent 实例
agent = acp.get_info("tool_calling").instance

# 运行优化
await optimize_agent_with_textgrad(
    agent=agent,
    task="Create a simple HTML Sokoban web mini-game",
    files=[],
    optimization_steps=3,
    optimizer_model="gpt-4o",
    log_dir=config.workdir
)
```

## 使用方式

### 方式1：便捷函数（推荐）

适合快速使用，自动处理大部分配置：

```python
from src.optimizers.textgrad_optimizer import optimize_agent_with_textgrad

# 在异步函数中调用
async def main():
    # ... 初始化 Agent、环境、工具等 ...
    
    agent = acp.get_info("tool_calling").instance
    
    # 执行优化
    optimizer = await optimize_agent_with_textgrad(
        agent=agent,
        task="Your task description here",
        files=[],  # 可选的文件列表
        optimization_steps=3,  # 优化迭代次数
        optimizer_model="gpt-4o",  # 用于优化的模型
        log_dir=config.workdir  # 日志保存目录
    )
    
    # 可以使用返回的 optimizer 对象访问优化后的变量
    optimized_vars = optimizer.get_optimized_variables()
    
    # 使用优化后的提示词运行 Agent
    result = await agent.ainvoke(task="Your task", files=[])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**完整示例**：参考 `examples/run_tool_calling_agent_optimized.py`

### 方式2：使用类（更灵活）

适合需要更多控制或自定义优化的场景：

```python
from src.optimizers.textgrad_optimizer import TextGradOptimizer

async def main():
    # ... 初始化 Agent ...
    
    agent = acp.get_info("tool_calling").instance
    
    # 创建优化器实例
    optimizer = TextGradOptimizer(
        agent=agent,
        log_dir=config.workdir  # 日志保存目录
    )
    
    # 执行优化
    await optimizer.optimize(
        task="Your task description here",
        files=[],  # 可选的文件列表
        optimization_steps=3,  # 优化迭代次数
        optimizer_model="gpt-4o"  # 用于优化的模型名称或引擎对象
    )
    
    # 获取优化后的变量
    optimized_vars = optimizer.get_optimized_variables()
    
    # 访问特定优化后的变量
    for tg_var in optimized_vars:
        print(f"变量描述: {tg_var.role_description}")
        print(f"优化后的值: {tg_var.value}")
    
    # 使用优化后的提示词运行 Agent
    result = await agent.ainvoke(task="Your task", files=[])

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## 参数说明

### `optimize_agent_with_textgrad()` 函数参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `agent` | Agent 实例 | ✅ | 要优化的 Agent 对象，必须包含 `prompt_manager` 和 `ainvoke` 方法 |
| `task` | `str` | ✅ | Agent 要执行的任务描述 |
| `files` | `List[str]` | ❌ | 附件文件路径列表，默认为 `[]` |
| `optimization_steps` | `int` | ❌ | 优化迭代次数，默认为 `3` |
| `optimizer_model` | `str` 或引擎对象 | ❌ | 用于优化的 LLM 模型，默认为 `"gpt-4o"` |
| `log_dir` | `str` | ❌ | 日志保存目录，如果为 `None` 则使用 Agent 的 `workdir` |

### `TextGradOptimizer.optimize()` 方法参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `task` | `str` | ✅ | Agent 要执行的任务描述 |
| `files` | `List[str]` | ❌ | 附件文件路径列表，默认为 `[]` |
| `optimization_steps` | `int` | ❌ | 优化迭代次数，默认为 `3` |
| `optimizer_model` | `str` 或引擎对象 | ❌ | 用于优化的 LLM 模型，默认为 `"gpt-4o"` |

### `TextGradOptimizer` 类方法

#### `extract_optimizable_variables()`

从 Agent 的**所有 prompt 对象**中提取标记为可优化的变量（`require_grad=True`），并转换为 TextGrad 变量格式。

**返回**：`Tuple[List[tg.Variable], Dict, Dict]` - (可优化的变量列表, 变量映射字典, prompt对象映射字典)
- `var_mapping`: `tg_var -> orig_var` - TextGrad 变量到原始变量的映射
- `prompt_mapping`: `tg_var -> prompt_obj` - TextGrad 变量到包含它的 prompt 对象的映射

**支持多 prompt 对象**：会自动从 `system_prompt` 和 `agent_message_prompt` 等所有 prompt 对象中提取变量。

#### `clear_prompt_caches()`

清除所有包含被优化变量的 prompt 对象的缓存，确保下次调用时使用更新后的变量值。

**参数**：
- `tg_vars` (可选): 要清除缓存的变量列表，如果为 `None` 则清除所有已记录的变量对应的 prompt

**工作原理**：
- 根据 `prompt_mapping` 找到每个变量所属的 prompt 对象
- 设置 `prompt_obj.message = None` 清除缓存
- 下次 Agent 调用 `get_message()` 时会自动重新渲染（见"重新加载的精确位置"章节）

#### `get_optimized_variables()`

获取优化后的变量列表。

**返回**：`List[tg.Variable]` - 优化后的变量列表

## 工作原理

### 1. 变量提取阶段

优化器会从 Agent 的**所有 prompt 对象**中递归查找所有标记为 `require_grad=True` 的变量，并将它们转换为 TextGrad 的 `Variable` 对象。

#### 1.1 支持的 Prompt 对象

优化器会自动从以下 prompt 对象中提取变量：

- **`system_prompt`**：系统提示词（`SystemPrompt` 类）
- **`agent_message_prompt`**：Agent 消息提示词（`AgentMessagePrompt` 类）
- **扩展支持**：可以通过修改 `find_prompt_objects_with_variables()` 方法添加更多 prompt 对象

#### 1.2 变量提取机制

- 使用 `extract_optimizable_variables()` 方法递归遍历所有 prompt 对象的 `prompt` 属性（Variable 树）
- 自动记录每个变量属于哪个 prompt 对象（存储在 `prompt_mapping` 中）
- 支持优化来自不同 prompt 对象的变量，优化器会统一管理并清除对应缓存

### 2. 优化循环（迭代步骤）

每次优化迭代包含以下步骤：

1. **同步提示词**：将 TextGrad 变量中的优化值同步回原始 Agent 变量，并**清除所有相关 prompt 对象的缓存**以确保使用新的值
2. **执行 Agent**：使用当前提示词运行 Agent，获取执行结果（此时会重新渲染 prompt，使用更新后的变量值）
3. **计算损失**：基于 Agent 执行结果和任务目标计算损失（使用 LLM 评估）
4. **生成梯度**：将损失反馈作为梯度添加到提示词变量（手动连接，因为 Agent 是黑盒）
5. **更新提示词**：使用 TextGrad 优化器（`TextualGradientDescent`）基于梯度更新提示词
6. **同步优化值**：将优化后的提示词值同步回原始 Agent 变量，并**清除所有相关 prompt 对象的缓存**

### 3. 优化机制说明

⚠️ **重要**：当前实现采用**黑盒优化**方式：

- Agent 的内部 LLM 调用（使用 LangChain）与 TextGrad 变量之间**没有直接的计算图连接**
- 优化器通过手动将 Agent 执行结果的损失反馈作为梯度添加到提示词变量
- 这是一种基于输入-输出反馈的优化方式，而不是通过推理过程的梯度传递

### 4. Prompt 缓存机制与重新加载

**重要**：所有 Prompt 类（`SystemPrompt`, `AgentMessagePrompt` 等）都使用缓存机制来避免重复渲染：

#### 4.1 缓存工作原理

- `Prompt.get_message(reload=False)` 会返回缓存的 `Message` 对象，不会重新渲染
- 如果 `message` 为 `None`，会重新渲染 prompt（调用 `prompt.render(modules)`）

#### 4.2 重新加载的精确位置

优化器更新变量值后，会清除所有相关 prompt 对象的缓存。重新加载发生在以下位置：

1. **Agent 执行流程**：
   ```
   Agent.ainvoke() 
   → Agent._get_messages() 
   → prompt_manager.get_system_message(reload=False) 或 get_agent_message(reload=True)
   → SystemPrompt.get_message(reload=False)
   ```

2. **Prompt 重新渲染时机**（在 `SystemPrompt.get_message()` 中，第 51-58 行）：
   ```python
   if not reload and self.message is not None:
       return self.message  # 返回缓存
   # 如果 message 为 None（缓存已清除），执行以下代码重新渲染：
   prompt_str = self.prompt.render(modules)  # ← 这里使用更新后的变量值
   self.message = SystemMessage(content=prompt_str, cache=True)
   ```

#### 4.3 优化器的缓存清除机制

- 优化器通过 `clear_prompt_caches()` 方法自动清除所有包含被优化变量的 prompt 对象的缓存
- **支持多 prompt 对象**：会自动从 `system_prompt` 和 `agent_message_prompt` 中查找变量并清除对应缓存
- **无需重新创建 Agent**：优化后的 prompt 会在 Agent 下次调用时自动生效

### 5. 日志记录

每次优化都会生成详细的日志文件，保存在 `log_dir/optimization_logs/` 目录下，包含：
- 每次迭代的 Agent 执行结果
- 损失计算详情
- 最终优化变量的摘要

## 注意事项

### 1. 提示词变量标记

确保在提示词模板中正确标记需要优化的变量：

```python
{
    "name": "agent_context_rules",
    "type": "system_prompt_module",
    "description": "Agent 上下文规则",
    "require_grad": True,  # ✅ 设置为 True 才会被优化
    "template": None,
    "variables": AGENT_CONTEXT_RULES
}
```

### 2. 模型选择

- **Agent 执行模型**：使用配置中的模型（如 `gpt-4.1`）
- **优化器模型**：建议使用较强的模型（如 `gpt-4o`），以确保优化器 LLM 能够：
  - 正确理解损失反馈
  - 生成高质量的提示词改进建议
  - 遵循输出格式要求（`<IMPROVED_VARIABLE>` 标签）

### 3. 优化步骤数

- **太少（1-2 步）**：可能无法充分优化
- **太多（>5 步）**：可能过度优化或成本过高
- **推荐**：3-4 步，根据任务复杂度调整

### 4. 错误处理

如果优化器步骤失败（通常是因为 LLM 未遵循输出格式），可以：
- 使用更强的模型（如 `gpt-4o`）
- 增加重试逻辑
- 检查优化器约束条件是否足够明确

### 5. 成本考虑

每次优化迭代都会调用：
- Agent 执行：使用 Agent 的模型执行任务
- 损失计算：使用优化器模型评估结果
- 提示词更新：使用优化器模型生成改进建议

请注意 API 调用成本，特别是在使用 GPT-4 系列模型时。

### 6. 异步执行

所有优化相关的方法都是异步的，必须在 `async` 函数中使用 `await` 调用。

## 完整示例

参考项目根目录下的 `examples/run_tool_calling_agent_optimized.py` 查看完整的集成示例。

## 更多信息

- TextGrad 项目文档：请参考 `textgrad/` 目录下的文档
- Agent 框架文档：请参考 `src/agents/` 目录下的文档
- 提示词模板：请参考 `src/agents/prompts/templates/` 目录下的模板文件

