import os
import asyncio
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# 1) 定义工具（支持 @tool / BaseTool / dict schema）
@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

tools = [add]

# 2) 准备一个支持“工具调用”的聊天模型
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    ).bind_tools(tools)

# 3) 一行创建 ReAct 代理图（v2 是默认，工具逐个执行，支持 Send API）
graph = create_react_agent(
    model=llm,
    tools=tools,
    prompt="你是一个严谨的助手，能在需要时使用工具再回答。"
)

# 4) 调用（一次）
result = graph.invoke({"messages": [("user", "12+30等于几？")]})
print(result["messages"][-1].content)

# 5) 流式事件（观察工具调用与思考过程的事件流）
async def run():
    async for ev in graph.astream_events(
        {"messages": [("user", "再算 7+8")]},
        version="v2"
    ):
        print(ev["event"], ev.get("data", ""))
        
asyncio.run(run())
