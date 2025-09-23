"""Python interpreter tool for executing Python code."""

import asyncio
from typing import Optional, Dict, Any, List, Type
from pydantic import BaseModel, Field
from typing import Type, Dict, Any

from src.tools.default_tools.executor import LocalPythonExecutor, BASE_BUILTIN_MODULES, BASE_PYTHON_TOOLS
from src.tools.protocol.tool import BaseTool
from src.tools.protocol.types import ToolResponse
from src.tools.protocol import tcp

_PYTHON_INTERPRETER_TOOL_DESCRIPTION = """Execute Python code and return the output.
Use this tool to run Python scripts, perform calculations, or execute any Python code.
The tool provides a safe execution environment with access to standard Python libraries.
"""

class PythonInterpreterArgs(BaseModel):
    code: str = Field(description="Python code to execute")

@tcp.tool()
class PythonInterpreterTool(BaseTool):
    """A tool that can execute Python code."""
    
    name: str = "python_interpreter"
    type: str = "Code Execution"
    description: str = _PYTHON_INTERPRETER_TOOL_DESCRIPTION
    args_schema: Type[BaseModel] = PythonInterpreterArgs
    metadata: Dict[str, Any] = {}
    
    authorized_imports: Optional[List[str]] = None
    base_python_tools: Optional[List[str]] = None
    python_evaluator: Optional["LocalPythonExecutor"] = None
    
    def __init__(self, **kwargs):
        """A tool that can execute Python code."""
        super().__init__(**kwargs)

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
        
    async def _arun(self, code: str) -> ToolResponse:
        try:
            self.python_evaluator.state = {}
            code_output = self.python_evaluator(code)
            output = f"Stdout:\n{code_output.logs}\nOutput: {str(code_output.output)}"
            return ToolResponse(content=output)
        
        except Exception as e:
            return ToolResponse(content=f"Error: {str(e)}")
        
    def _run(self, code: str) -> ToolResponse:
        """Execute Python code synchronously (fallback)."""
        try:
            # Run the async version in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(code))
            finally:
                loop.close()
        except Exception as e:
            return ToolResponse(content=f"Error in synchronous execution: {str(e)}")
