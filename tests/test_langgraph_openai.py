from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.utils.function_calling import format_tool_to_openai_function
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from typing import Type

load_dotenv()

class AddNumbersArgs(BaseModel):
    a: int = Field(description="第一个数")
    b: int = Field(description="第二个数")

class AddNumbersTool(BaseTool):
    """两个数相加"""
    name: str = "add_numbers"
    description: str = "两个数相加"
    args_schema: Type[AddNumbersArgs] = AddNumbersArgs

    def _run(self, args: AddNumbersArgs) -> int:
        """两个数相加"""
        return args.a + args.b
    

llm = ChatOpenAI(model="gpt-4o")  # 或者 gpt-4o, gpt-3.5-turbo 等

print(format_tool_to_openai_function(AddNumbersTool()))
exit()

# 把工具传进去
llm_with_tools = llm.bind_tools([add_numbers])

response = llm_with_tools.invoke("帮我把 3 和 5 加起来")
print(response)