"""Done tool for indicating that the task has been completed."""
from src.tool.types import Tool, ToolResponse

_DONE_TOOL_DESCRIPTION = """Done tool for indicating that the task has been completed.
Use this tool to signal that a task or subtask has been finished.
Provide a brief summary of what was accomplished.
"""

class DoneTool(Tool):
    """A tool for indicating that the task has been completed."""

    name: str = "done"
    description: str = _DONE_TOOL_DESCRIPTION
    enabled: bool = True
    
    def __init__(self, **kwargs):
        """A tool for indicating that the task has been completed."""
        super().__init__(**kwargs)

    async def __call__(self, result: str, **kwargs) -> ToolResponse:
        """
        Indicate that the task has been completed.

        Args:
            result (str): Summary of the accomplished task.
        """
        return ToolResponse(success=True, message=f"✅ Task completed: {result}")