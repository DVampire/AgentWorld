"""Unified file operation tool for reading, writing, deleting, modifying files and searching with grep."""

import asyncio
import os
import re
from typing import Optional, Dict, Any
from langchain.tools import BaseTool
from pydantic import Field


_FILE_TOOL_DESCRIPTION = """Unified file operation tool that supports multiple file operations.
Use absolute paths for all file operations to avoid path-related issues.

Available operations:
1. READ: Read file content with line numbers
   - file_path: absolute path to file
   - start_line: starting line number (optional, 1-indexed)
   - end_line: ending line number (optional, 1-indexed)

2. WRITE: Write content to file
   - file_path: absolute path to file
   - content: content to write
   - mode: 'w' for overwrite, 'a' for append (default: 'w')

3. DELETE: Delete a file
   - file_path: absolute path to file to delete

4. MODIFY: Modify a specific line in file
   - file_path: absolute path to file
   - line_number: line number to modify (1-indexed)
   - new_content: new content for the line

5. GREP: Search for patterns in file
   - file_path: absolute path to file
   - pattern: search pattern (regex supported)
   - case_sensitive: whether search is case sensitive (default: True)

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "read", "file_path": "/path/to/file.txt", "start_line": 1, "end_line": 10}
"""


class FileTool(BaseTool):
    """Unified tool for all file operations."""
    
    name: str = "file"
    description: str = _FILE_TOOL_DESCRIPTION
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def _arun(self, input_json: str) -> str:
        """Execute file operation asynchronously based on input JSON."""
        try:
            import json
            params = json.loads(input_json)
            operation = params.get("operation", "").upper()
            
            if operation == "READ":
                return await self._read_file(params)
            elif operation == "WRITE":
                return await self._write_file(params)
            elif operation == "DELETE":
                return await self._delete_file(params)
            elif operation == "MODIFY":
                return await self._modify_file(params)
            elif operation == "GREP":
                return await self._grep_file(params)
            else:
                return f"Error: Unknown operation '{operation}'. Supported operations: READ, WRITE, DELETE, MODIFY, GREP"
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input. Please provide valid JSON with 'operation' and parameters."
        except Exception as e:
            return f"Error executing file operation: {str(e)}"
    
    async def _read_file(self, params: Dict[str, Any]) -> str:
        """Read file content with line numbers."""
        try:
            file_path = params.get("file_path")
            start_line = params.get("start_line")
            end_line = params.get("end_line")
            
            if not file_path:
                return "Error: 'file_path' is required for READ operation"
            
            if not os.path.isabs(file_path):
                return "Error: File path must be absolute"
            
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"
            
            async with asyncio.Lock():
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                total_lines = len(lines)
                
                # Determine line range
                start = start_line - 1 if start_line else 0
                end = end_line if end_line else total_lines
                
                # Validate line numbers
                if start < 0 or end > total_lines or start >= end:
                    return f"Error: Invalid line range. File has {total_lines} lines."
                
                # Format output with line numbers
                result_lines = []
                for i in range(start, end):
                    line_num = i + 1
                    content = lines[i].rstrip('\n')
                    result_lines.append(f"{line_num:4d}: {content}")
                
                result = f"File: {file_path}\nLines {start + 1}-{end} of {total_lines}:\n"
                result += "\n".join(result_lines)
                
                return result
                
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    async def _write_file(self, params: Dict[str, Any]) -> str:
        """Write content to file."""
        try:
            file_path = params.get("file_path")
            content = params.get("content")
            mode = params.get("mode", "w")
            
            if not file_path:
                return "Error: 'file_path' is required for WRITE operation"
            if content is None:
                return "Error: 'content' is required for WRITE operation"
            
            if not os.path.isabs(file_path):
                return "Error: File path must be absolute"
            
            if mode not in ["w", "a"]:
                return "Error: Mode must be 'w' (overwrite) or 'a' (append)"
            
            async with asyncio.Lock():
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                with open(file_path, mode, encoding='utf-8') as f:
                    f.write(content)
                
                action = "overwritten" if mode == "w" else "appended to"
                return f"Successfully {action} file: {file_path}"
                
        except Exception as e:
            return f"Error writing file: {str(e)}"
    
    async def _delete_file(self, params: Dict[str, Any]) -> str:
        """Delete a file."""
        try:
            file_path = params.get("file_path")
            
            if not file_path:
                return "Error: 'file_path' is required for DELETE operation"
            
            if not os.path.isabs(file_path):
                return "Error: File path must be absolute"
            
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"
            
            async with asyncio.Lock():
                os.remove(file_path)
                return f"Successfully deleted file: {file_path}"
                
        except Exception as e:
            return f"Error deleting file: {str(e)}"
    
    async def _modify_file(self, params: Dict[str, Any]) -> str:
        """Modify a specific line in file."""
        try:
            file_path = params.get("file_path")
            line_number = params.get("line_number")
            new_content = params.get("new_content")
            
            if not file_path:
                return "Error: 'file_path' is required for MODIFY operation"
            if line_number is None:
                return "Error: 'line_number' is required for MODIFY operation"
            if new_content is None:
                return "Error: 'new_content' is required for MODIFY operation"
            
            if not os.path.isabs(file_path):
                return "Error: File path must be absolute"
            
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"
            
            if line_number < 1:
                return "Error: Line number must be 1 or greater"
            
            async with asyncio.Lock():
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if line_number > len(lines):
                    return f"Error: Line {line_number} does not exist. File has {len(lines)} lines."
                
                # Modify the specified line (convert to 0-indexed)
                lines[line_number - 1] = new_content + '\n'
                
                # Write back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(lines)
                
                return f"Successfully modified line {line_number} in file: {file_path}"
                
        except Exception as e:
            return f"Error modifying file: {str(e)}"
    
    async def _grep_file(self, params: Dict[str, Any]) -> str:
        """Search for patterns in file."""
        try:
            file_path = params.get("file_path")
            pattern = params.get("pattern")
            case_sensitive = params.get("case_sensitive", True)
            
            if not file_path:
                return "Error: 'file_path' is required for GREP operation"
            if not pattern:
                return "Error: 'pattern' is required for GREP operation"
            
            if not os.path.isabs(file_path):
                return "Error: File path must be absolute"
            
            if not os.path.exists(file_path):
                return f"Error: File not found: {file_path}"
            
            async with asyncio.Lock():
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Compile regex pattern
                flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    regex = re.compile(pattern, flags)
                except re.error as e:
                    return f"Error: Invalid regex pattern: {str(e)}"
                
                # Search for matches
                matches = []
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        content = line.rstrip('\n')
                        matches.append(f"{i:4d}: {content}")
                
                if not matches:
                    return f"No matches found for pattern '{pattern}' in file: {file_path}"
                
                result = f"Found {len(matches)} matches for pattern '{pattern}' in file: {file_path}:\n"
                result += "\n".join(matches)
                
                return result
                
        except Exception as e:
            return f"Error searching file: {str(e)}"
    
    def _run(self, input_json: str) -> str:
        """Execute file operation synchronously."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(input_json))
            finally:
                loop.close()
        except Exception as e:
            return f"Error in synchronous execution: {str(e)}"
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "type": "file"
        }

