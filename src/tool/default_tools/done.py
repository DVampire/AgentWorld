"""Done tool for indicating that the task has been completed."""
from typing import Dict, Any
from pydantic import Field
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.registry import TOOL

_DONE_TOOL_DESCRIPTION = """Done tool for indicating that the task has been completed.
Use this tool to signal that a task or subtask has been finished.
Provide the `result` and `reasoning` of the task in the result and reasoning parameters.

Args:
- result (str): The result of the task completion.
- reasoning (str): The analysis or explanation of the task completion.

Example: {"name": "done_tool", "args": {"reasoning": "The task has been completed successfully.","result": "The task has been completed."}}.
"""

@TOOL.register_module(force=True)
class DoneTool(Tool):
    """A tool for indicating that the task has been completed."""

    name: str = "done_tool"
    description: str = _DONE_TOOL_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")
    
    def __init__(self, require_grad: bool = False, **kwargs):
        """A tool for indicating that the task has been completed."""
        super().__init__(require_grad=require_grad, **kwargs)

    async def __call__(self, 
                       reasoning: str,
                       result: str,
                       **kwargs) -> ToolResponse:
        """
        Indicate that the task has been completed.

        Args:
            reasoning (str): The reasoning of the task completion. Must be provided.
            result (str): The result of the task completion. Must be provided.
        """
        # Convert to string in case LLM returns non-string types
        if reasoning is None or reasoning == "":
            reasoning = "No reasoning provided"
        else:
            reasoning = str(reasoning)
        if result is None or result == "":
            result = "No result provided"
        else:
            result = str(result)
        return ToolResponse(success=True, message=result, extra=ToolExtra(data={"reasoning": reasoning, "result": result}))