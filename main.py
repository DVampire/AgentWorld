from langchain.tools import BaseTool
from pydantic import Field, BaseModel
from typing import Type


class TestArgs(BaseModel):
    name: str = Field(description="The name of the test")
    age: int = Field(description="The age of the test")

class Test(BaseTool):
    name: str = "test"
    description: str = "The description of the test"
    args_schema: Type[BaseModel] = TestArgs
    
    def _run(self, name: str, age: int) -> str:
        return f"Hello, {name}! You are {age} years old."
    
    async def _arun(self, name: str, age: int) -> str:
        return f"Hello, {name}! You are {age} years old."
    
test = Test()

test.metadata = {"type": "test"}

print(test.invoke(input={"name": "John", "age": 20}))

print(test.metadata)