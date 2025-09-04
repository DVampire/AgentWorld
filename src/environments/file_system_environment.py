"""File System Environment for AgentWorld - provides file system operations as an environment."""

import asyncio
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from inspect import cleandoc

from src.filesystem.file_system import FileSystem
from src.logger import logger
from src.registry import ENVIRONMENTS


@ENVIRONMENTS.register_module(force=True)
class FileSystemEnvironment:
    """File System Environment that provides file operations as an environment interface."""
    
    def __init__(
        self,
        base_dir: str | Path,
        create_default_files: bool = True,
        max_file_size: int = 1024 * 1024,  # 1MB
    ):
        """
        Initialize the file system environment.
        
        Args:
            base_dir: Base directory for the file system
            create_default_files: Whether to create default files (todo.md)
            max_file_size: Maximum file size in bytes
            allowed_extensions: List of allowed file extensions (None for all supported)
        """
        self.base_dir = Path(base_dir) if isinstance(base_dir, str) else base_dir
        self.max_file_size = max_file_size
        
        # Initialize file system
        self.file_system = FileSystem(
            base_dir=self.base_dir,
            create_default_files=create_default_files
        )
        
        logger.info(f"| 🗂️ File System Environment initialized at: {self.base_dir}")
        logger.info(f"| 📁 Available files: {self.file_system.list_files()}")
    
    def get_state(self):
        """Get current environment state."""
        todo_contents = self.file_system.get_todo_contents()
        description = self.file_system.describe()
        state = dict(
            description=description,
            todo_contents=todo_contents,
        )
        return state
        
    
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Returns:
            Tuple of (state, info)
        """
        # Reset file system to initial state
        self.file_system = FileSystem(
            base_dir=self.base_dir,
            create_default_files=True
        )
        
        # done
        done = False
        
        # Get state and info
        state = self.get_state()
        state.update(dict(
            state="Initial the file system environment successfully.",
        ))
        info = dict(done=done)
        
        logger.info("| 🔄 File System Environment reset to initial state")
        return state, info
    
    def step(self, 
             operation: str,
             filename: str,
             content: Optional[str] = None,
             old_string: Optional[str] = None,
             new_string: Optional[str] = None,
             ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action dictionary containing operation and parameters
            
        Returns:
            Tuple of (state, reward, terminated, truncated, info)
        """
       
        # Execute operation
        operation_result = asyncio.run(self._execute_operation(
            operation, filename, content, old_string, new_string
        ))
        
        # Calculate reward (simple: +1 for success, -1 for failure)
        success = operation_result.get("success", False)
        reward = 1.0 if success else -1.0
        
        state = self.get_state()
        state.update(dict(
            state=operation_result.get("state", ""),
        ))
        
        # Environment doesn't naturally terminate, so always False
        done = False
        truncated = False
        
        # Create info with operation result
        info = dict(done=done)
        info.update(operation_result)
        
        return state, reward, done, truncated, info
    
    async def _execute_operation(
        self,
        operation: str,
        filename: str,
        content: Optional[str] = None,
        old_string: Optional[str] = None,
        new_string: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a file system operation.
        
        Args:
            operation: Operation name
            filename: File name
            content: File content (for write/append)
            old_string: Old string (for replace)
            new_string: New string (for replace)
            
        Returns:
            Dictionary with operation result
        """
        try:
            if operation == "read":
                return await self._read_file(filename)
            elif operation == "write":
                return await self._write_file(filename, content)
            elif operation == "append":
                return await self._append_file(filename, content)
            elif operation == "replace":
                return await self._replace_file_str(filename, old_string, new_string)
            else:
                return dict(
                    success=False,
                    operation=operation,
                    filename=filename,
                    state=f"Unknown operation: {operation}",
                )
        
        except Exception as e:
            logger.error(f"| ❌ Error executing operation '{operation}': {e}")
            return dict(
                success=False,
                operation=operation,
                filename=filename,
                state=f"Error executing operation: {str(e)}",
            )
    
    async def _read_file(self, filename: str) ->Dict[str, Any]:
        """Read file operation."""
        result = await self.file_system.read_file(filename)
        success = not result.startswith("Error:")
        
        state = cleandoc(f"""
        Success: {success}
        Operation: read
        File: {filename}
        Result: {result}
        """).strip()
        
        res = dict(
            success=success,
            operation="read",
            filename=filename,
            state=state,
        )
        return res
    
    async def _write_file(self, filename: str, content: str) -> Dict[str, Any]:
        """Write file operation."""
        # Check file size constraint
        if len(content) > self.max_file_size:
            return dict(
                success=False,
                operation="write",
                filename=filename,
                state=f"File size exceeds maximum allowed size of {self.max_file_size} bytes",
            )
        
        result = await self.file_system.write_file(filename, content)
        success = not result.startswith("Error:")
        
        state = cleandoc(f"""
        Success: {success}
        Operation: write
        File: {filename}
        Result: {result}
        """).strip()
        
        res = dict(
            success=success,
            operation="write",
            filename=filename,
            state=state,
        )
        return res
    
    async def _append_file(self, filename: str, content: str) -> Dict[str, Any]:
        """Append file operation."""
        result = await self.file_system.append_file(filename, content)
        success = not result.startswith("Error:")
        
        state = cleandoc(f"""
        Success: {success}
        Operation: append
        File: {filename}
        Result: {result}
        """).strip()
        
        res = dict(
            success=success,
            operation="append",
            filename=filename,
            state=state,
        )
        return res
    
    async def _replace_file_str(self, filename: str, old_str: str, new_str: str) -> Dict[str, Any]:
        """Replace string in file operation."""
        result = await self.file_system.replace_file_str(filename, old_str, new_str)
        success = not result.startswith("Error:")
        
        state = cleandoc(f"""
        Success: {success}
        Operation: replace
        File: {filename}
        Old String: {old_str}
        New String: {new_str}
        Result: {result}
        """).strip()
        
        res = dict(
            success=success,
            operation="replace",
            filename=filename,
            state=state,
        )
        return res
    
    def close(self):
        """Close the environment and cleanup resources."""
        logger.info("| 🔒 File System Environment closed")
