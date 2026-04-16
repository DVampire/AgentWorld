from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import tool
from langchain import hub
import asyncio

# 定义工具
@tool
def calculate(expression: str) -> str:
    """计算一个算式"""
    return str(eval(expression))

@tool
def get_time() -> str:
    """返回当前时间"""
    import datetime
    return str(datetime.datetime.now())

tools = [calculate, get_time]

# 模型
llm = ChatOpenAI(model="gpt-4o-mini")

# prompt
prompt = hub.pull("hwchase17/openai-functions-agent")

# agent + executor
agent = create_openai_functions_agent(llm, tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

async def main():
    async for step in executor.astream({"input": "帮我算 (10+5)*3，然后告诉我现在的时间"}):
        print("---- step ----")
        print(step)

asyncio.run(main())
