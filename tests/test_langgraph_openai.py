from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Type
import asyncio

load_dotenv()

class AddNumbersArgs(BaseModel):
    a: int = Field(description="第一个数")
    b: int = Field(description="第二个数")

class AddNumbersTool(BaseTool):
    """两个数相加"""
    name: str = "add_numbers"
    description: str = "两个数相加"
    args_schema: Type[AddNumbersArgs] = AddNumbersArgs

    def _run(self, a: int, b: int) -> int:
        return a + b

    async def _arun(self, a: int, b: int) -> int:
        return a + b
    

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 让模型知道有哪些工具
llm_with_tools = llm.bind_tools([AddNumbersTool()])

# 第一步：问模型
response = llm_with_tools.invoke("帮我把 3 和 5 加起来")
print("模型输出:", response)

# 第二步：解析模型请求
tool_call = response.tool_calls[0]
tool_args = tool_call["args"]

# 第三步：执行工具
result = asyncio.run(AddNumbersTool().ainvoke(input=tool_args))
print("工具执行结果:", result)