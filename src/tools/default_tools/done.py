"""Done tool for indicating that the task has been completed."""
from pydantic import BaseModel, Field
from typing import Type, Dict, Any

from src.tools.protocol.tool import BaseTool
from src.tools.protocol.types import ToolResponse
from src.tools.protocol import tcp

_DONE_TOOL_DESCRIPTION = """Done tool for indicating that the task has been completed.
Use this tool to signal that a task or subtask has been finished.
Provide a brief summary of what was accomplished.
"""

class DoneToolArgs(BaseModel):
    result: str = Field(description="The result of the task")

@tcp.tool()
class DoneTool(BaseTool):
    """A tool for indicating that the task has been completed."""
    
    name: str = "done"
    type: str = "Done"
    description: str = _DONE_TOOL_DESCRIPTION
    args_schema: Type[BaseModel] = DoneToolArgs
    metadata: Dict[str, Any] = {}
    
    def __init__(self, **kwargs):
        """A tool for indicating that the task has been completed."""
        super().__init__(**kwargs)
    
    async def _arun(self, result: str) -> ToolResponse:
        """Indicate that the task has been completed."""
        return ToolResponse(content=f"✅ Task completed: {result}")
    
    def _run(self, result: str) -> ToolResponse:
        """Indicate that the task has been completed."""
        return ToolResponse(content=f"✅ Task completed: {result}")