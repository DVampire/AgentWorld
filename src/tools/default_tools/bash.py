"""Bash tool for executing shell commands."""

import asyncio
import subprocess
import shlex
from typing import Optional, Dict, Any
from langchain.tools import BaseTool
from pydantic import Field

_BASH_TOOL_DESCRIPTION = """Execute bash commands in the shell. 
Use this tool to run system commands, scripts, or any bash operations. 
Be careful with commands that modify the system or require elevated privileges. 
For file operations, ALWAYS use absolute paths to avoid path-related issues. 
Input should be a valid bash command string.
"""


class BashTool(BaseTool):
    """A tool for executing bash commands asynchronously."""
    
    name: str = "bash"
    description: str = _BASH_TOOL_DESCRIPTION
    
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for command execution"
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def _arun(self, command: str) -> str:
        """Execute a bash command asynchronously."""
        try:
            # Sanitize the command
            if not command.strip():
                return "Error: Empty command provided"
            
            # Parse command and arguments
            args = shlex.split(command)
            if not args:
                return "Error: Invalid command format"
            
            # Create process with timeout
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return f"Error: Command timed out after {self.timeout} seconds"
            
            # Decode output
            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()
            
            # Prepare result
            result = []
            if stdout_str:
                result.append(f"STDOUT:\n{stdout_str}")
            if stderr_str:
                result.append(f"STDERR:\n{stderr_str}")
            
            exit_code = process.returncode
            if exit_code != 0:
                result.append(f"Exit code: {exit_code}")
            
            return "\n\n".join(result) if result else f"Command completed with exit code: {exit_code}"
            
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def _run(self, command: str) -> str:
        """Execute a bash command synchronously (fallback)."""
        try:
            # Run the async version in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(command))
            finally:
                loop.close()
        except Exception as e:
            return f"Error in synchronous execution: {str(e)}"
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "timeout": self.timeout,
            "type": "bash"
        }
