"""Done tool for indicating that the task has been completed."""

import asyncio
import os
import re
from typing import Optional, Dict, Any, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


_DONE_TOOL_DESCRIPTION = """Done tool for indicating that the task has been completed."""

class DoneToolArgs(BaseModel):
    result: str = Field(description="The result of the task")


class DoneTool(BaseTool):
    """Done tool for indicating that the task has been completed."""
    
    name: str = "done"
    description: str = _DONE_TOOL_DESCRIPTION
    args_schema: Type[DoneToolArgs] = DoneToolArgs
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def _arun(self, result: str) -> str:
        """Indicate that the task has been completed."""
        return result
    
    def _run(self, result: str) -> str:
        """Indicate that the task has been completed."""
        return result
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get the tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema
        }