"""Done tool for indicating that the task has been completed."""
from typing import Dict, Any
from pydantic import Field
from src.tool.types import Tool, ToolResponse, ToolExtra
from typing import Optional
from src.registry import TOOL

_DONE_TOOL_DESCRIPTION = """Done tool for indicating that the task has been completed.
Use this tool to signal that a task or subtask has been finished.
Provide the `result` and `reasoning` of the task in the result and reasoning parameters.
"""

@TOOL.register_module(force=True)
class DoneTool(Tool):
    """A tool for indicating that the task has been completed."""

    name: str = "done"
    description: str = _DONE_TOOL_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")
    
    def __init__(self, require_grad: bool = False, **kwargs):
        """A tool for indicating that the task has been completed."""
        super().__init__(require_grad=require_grad, **kwargs)

    async def __call__(self, 
                       result: str,
                       reasoning: str = None,
                       **kwargs) -> ToolResponse:
        """
        Indicate that the task has been completed.

        Args:
            result (str): The result of the task completion.
            reasoning (str): The analysis or explanation of the task completion.
        """
        return ToolResponse(success=True, message=result, extra=ToolExtra(data={"result": result, "reasoning": reasoning}))