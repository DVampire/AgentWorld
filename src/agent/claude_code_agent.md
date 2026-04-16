# ClaudeCodeAgent 部署要求

## 依赖安装
```bash
pip install claude-agent-sdk  # 自带 claude CLI 二进制，无需单独安装
```

## 认证（二选一）

**方式 1：OAuth 登录（推荐）**
```bash
claude  # 首次运行会弹出浏览器登录，认证存储在 ~/.claude/
```

**方式 2：API Key**
```bash
export ANTHROPIC_API_KEY=sk-ant-xxx
```
或在 config 里指定：
```python
claude_code_agent.update(api_key="sk-ant-xxx")
```

## 注意
- 认证状态持久化在 `~/.claude/`，只需登录一次
- 默认模型 `claude-opus-4-6`，需要有对应模型权限
