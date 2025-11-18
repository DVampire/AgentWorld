# Report Tool 使用说明

## 📋 工具简介

Report Tool 是一个基于模板的投资银行风格 HTML 报告生成工具。它能够自动搜索最新新闻，分析市场动态，并生成包含交互式关系图谱、概率分析和完整市场分析的专业报告。

## ✨ 功能特点

- **自动新闻搜索**：使用 Tavily API 自动搜索与主题相关的最新新闻（默认最近7天）
- **关系图谱可视化**：自动提取实体关系，生成交互式关系图谱（使用 vis.js）
- **概率分析**：提供股票价格上涨/下跌概率分析及详细原因
- **新闻分析**：综合分析重大事件、公告、监管变化等
- **图谱解读**：深入解读关系图谱中的实体连接和网络洞察
- **专业模板**：使用投资银行风格的 HTML 模板，美观专业

## 🔧 环境配置

### 必需的环境变量

在 `.env` 文件中配置以下环境变量：

```bash
# OpenAI 兼容 API（必需）
# 可以使用以下任一变量名：
OPENAI_API_KEY=your_api_key_here
# 或
REPORT_ENGINE_API_KEY=your_api_key_here
# 或
DEEPSEEK_API_KEY=your_api_key_here

# API Base URL（可选，如果使用自定义 API）
OPENAI_API_BASE=https://api.openai.com/v1
# 或
REPORT_ENGINE_BASE_URL=https://your-api-base-url.com/v1

# 模型名称（可选，默认使用 gpt-4o-mini）
OPENAI_MODEL_NAME=gpt-4o-mini
# 或
REPORT_ENGINE_MODEL_NAME=gpt-4o-mini

# Tavily API Key（必需，用于新闻搜索）
TAVILY_API_KEY=your_tavily_api_key_here
```

### 获取 API Keys

1. **OpenAI API Key**：
   - 访问 https://platform.openai.com/api-keys
   - 创建新的 API key

2. **Tavily API Key**：
   - 访问 https://tavily.com/
   - 注册账号并获取 API key

## 🚀 使用方法

### 方法 1：通过工具系统调用（推荐）

```python
import asyncio
from pathlib import Path
import sys

# 添加项目根目录到路径
root = str(Path(__file__).resolve().parents[0])
sys.path.append(root)

from src.tools import tcp
from src.models import model_manager
from src.logger import logger

async def generate_report():
    # 初始化模型管理器
    await model_manager.initialize(use_local_proxy=False)
    
    # 初始化工具
    await tcp.initialize(["report"])
    
    # 生成报告
    query = "What is the latest news about Apple stock price and market analysis?"
    result = await tcp.ainvoke("report", {"query": query})
    
    print(result.message)
    if result.extra:
        print(f"报告路径: {result.extra.get('path')}")

# 运行
asyncio.run(generate_report())
```

### 方法 2：指定自定义输出路径

```python
result = await tcp.ainvoke("report", {
    "query": "Tesla stock analysis and recent market trends",
    "output_path": "workdir/my_reports/tesla_analysis.html"
})
```

### 方法 3：独立脚本执行

```bash
python src/tools/report/report.py "What is the latest news about Apple stock price?"
```

## 📝 参数说明

### ReportToolArgs

- **query** (str, 必需)
  - 描述：生成报告的主题或问题描述
  - 示例：`"What is the latest news about Apple stock price and market analysis?"`

- **output_path** (str, 可选)
  - 描述：指定报告保存的完整路径（包括文件名）
  - 默认：`workdir/base/reports/tool_report_{query}_{timestamp}.html`
  - 示例：`"workdir/my_reports/custom_report.html"`

## 📊 报告内容结构

生成的报告包含以下部分：

1. **报告头部**
   - 公司名称
   - 报告日期（自动设置为当前日期）

2. **市场关系图谱**（左侧）
   - 交互式可视化图谱
   - 显示实体（公司、人物、产品、事件等）及其关系
   - 支持拖拽、缩放、重置视图

3. **新闻分析与图谱解读**（右侧）
   - 新闻分析摘要
   - 关系图谱解读

4. **股票价格变动概率分析**
   - **上涨概率**（Upside Probability）
     - 概率值（0-100%）
     - 上涨原因摘要
     - 详细分析（可展开查看）
   - **下跌概率**（Downside Probability）
     - 概率值（0-100%）
     - 下跌原因摘要
     - 详细分析（可展开查看）

## 📁 输出文件

### 默认保存位置

```
workdir/base/reports/tool_report_{查询内容}_{时间戳}.html
```

### 文件命名规则

- 查询内容会被清理（只保留字母、数字、空格、横线、下划线）
- 空格替换为下划线
- 最多 50 个字符
- 时间戳格式：`YYYYMMDD_HHMMSS`

### 示例文件名

```
tool_report_What_is_the_latest_news_about_Apple_stock_price_an_20251118_101214.html
```

## 🎨 报告特性

### 关系图谱

- **节点颜色编码**：
  - 🔵 蓝色：公司
  - 🟠 橙色：人物
  - 🟢 绿色：产品
  - 🟣 紫色：事件
  - ⚪ 灰色：其他实体

- **交互功能**：
  - 鼠标悬停查看详细信息
  - 拖拽节点调整布局
  - 缩放视图
  - 重置视图按钮

### 概率分析卡片

- 点击 "View Details" 按钮展开详细分析
- 上涨概率显示为绿色
- 下跌概率显示为红色

## 📖 使用示例

### 示例 1：分析苹果股票

```python
query = "What is the latest news about Apple stock price and market analysis?"
result = await tcp.ainvoke("report", {"query": query})
```

### 示例 2：分析特斯拉股票趋势

```python
query = "Tesla stock analysis and recent market trends"
result = await tcp.ainvoke("report", {
    "query": query,
    "output_path": "workdir/reports/tesla_analysis.html"
})
```

### 示例 3：分析加密货币市场

```python
query = "Bitcoin and Ethereum market analysis, recent price movements and regulatory news"
result = await tcp.ainvoke("report", {"query": query})
```

## 🔍 完整示例代码

参考 `examples/test_report_tool.py` 文件，包含三种测试方式：

1. 直接工具调用
2. 自定义输出路径
3. 独立脚本执行说明

## ⚠️ 注意事项

1. **API 限制**：
   - Tavily API 有调用次数限制，请合理使用
   - OpenAI API 按 token 计费，长报告可能产生较高费用

2. **网络要求**：
   - 报告使用 CDN 加载 vis.js 库，需要网络连接
   - 如果离线使用，需要下载 vis.js 并修改模板

3. **数据准确性**：
   - 报告基于 LLM 生成，数据可能不完全准确
   - 建议结合其他数据源进行验证

4. **日期自动修正**：
   - 报告日期会自动设置为当前日期
   - 即使 LLM 返回错误日期，也会自动修正

## 🐛 故障排除

### 问题 1：ModuleNotFoundError: No module named 'openai'

**解决方案**：安装 OpenAI 包
```bash
pip install openai
```

### 问题 2：Missing TAVILY_API_KEY

**解决方案**：在 `.env` 文件中添加 `TAVILY_API_KEY`

### 问题 3：报告日期不正确

**解决方案**：代码已自动修正日期，如果仍有问题，检查系统时间

### 问题 4：关系图谱不显示

**解决方案**：
- 检查网络连接（需要加载 vis.js）
- 检查浏览器控制台是否有错误
- 确认报告数据中包含 `relationshipGraph` 字段

## 📚 相关文件

- 工具实现：`src/tools/report/report.py`
- HTML 模板：`src/tools/report/report_template.html`
- 测试示例：`examples/test_report_tool.py`

## 🔗 相关链接

- [Tavily API 文档](https://docs.tavily.com/)
- [OpenAI API 文档](https://platform.openai.com/docs)
- [vis.js 文档](https://visjs.github.io/vis-network/docs/network/)

## 📝 更新日志

- **2025-11-18**：初始版本
  - 基于模板的报告生成
  - 自动新闻搜索
  - 关系图谱可视化
  - 概率分析功能
  - 自动日期修正

