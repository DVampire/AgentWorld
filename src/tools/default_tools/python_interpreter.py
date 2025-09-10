"""Unified file operation tool for reading, writing, deleting, modifying files and searching with grep."""

import asyncio
import os
import re
from typing import Optional, Dict, Any, List, Callable, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, field_validator

from src.tools.default_tools.executor import LocalPythonExecutor, BASE_BUILTIN_MODULES, BASE_PYTHON_TOOLS


_PYTHON_INTERPRETER_TOOL_DESCRIPTION = """It is a tool that can execute python code.

Input format: JSON string with 'code' parameter.
Example: {"name": "python_interpreter", "args": {"code": "print('Hello, world!')"}}
"""

class PythonInterpreterArgs(BaseModel):
    code: str = Field(..., description="Python code to execute")

class PythonInterpreterTool(BaseTool):
    """It is a tool that can execute python code."""
    
    name: str = "python_interpreter"
    description: str = _PYTHON_INTERPRETER_TOOL_DESCRIPTION
    args_schema: Type[PythonInterpreterArgs] = PythonInterpreterArgs

    authorized_imports: Optional[List[str]] = None
    base_python_tools: Optional[List[str]] = None
    python_evaluator: Optional["LocalPythonExecutor"] = None

    model_config = {"arbitrary_types_allowed": True}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def model_post_init(self, __context):
        if self.authorized_imports is None:
            self.authorized_imports = list(BASE_BUILTIN_MODULES)
        else:
            self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.authorized_imports))

        if self.base_python_tools is None:
            self.base_python_tools = dict(BASE_PYTHON_TOOLS)

        if self.python_evaluator is None:
            self.python_evaluator = LocalPythonExecutor(
                additional_authorized_imports=self.authorized_imports,
            )
            self.python_evaluator.send_tools(self.base_python_tools)
        
    async def _arun(self, code: str) -> str:
        try:
            self.python_evaluator.state = {}
            code_output = self.python_evaluator(code)
            output = f"Stdout:\n{code_output.logs}\nOutput: {str(code_output.output)}"
            return output
        
        except Exception as e:
            return f"Error: {str(e)}"
        
    def _run(self, code: str) -> str:
        """Execute a bash command synchronously (fallback)."""
        try:
            # Run the async version in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(code))
            finally:
                loop.close()
        except Exception as e:
            return f"Error in synchronous execution: {str(e)}"
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "type": "python_interpreter"
        }
