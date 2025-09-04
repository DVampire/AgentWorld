"""File System Environment for AgentWorld - provides file system operations as an environment."""

import asyncio
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from inspect import cleandoc

from sympy import N

from src.filesystem.file_system import FileSystem
from src.logger import logger
from src.registry import ENVIRONMENTS
from src.utils import assemble_project_path


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
        self.base_dir = Path(assemble_project_path(str(base_dir)))
        self.max_file_size = max_file_size
        
        # Initialize file system
        self.file_system = FileSystem(
            base_dir=self.base_dir,
            create_default_files=create_default_files
        )
        
        logger.info(f"| 🗂️ File System Environment initialized at: {self.base_dir}")

        self.state = None
        self.info = None
        self.done = None
        
    
    async def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
        state = dict(
            prompt=await self.file_system.describe(),
        )
        info = dict(done=done)
        
        self.state = state
        self.info = info
        self.done = done
        
        logger.info("| 🔄 File System Environment reset to initial state")
        return state, info
    
    async def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.
        
        Args:
            action: Action dictionary containing operation type and parameters
                   Supported action types:
                   - file_operations: File-related operations (read, write, delete, copy, move, rename, etc.)
                   - directory_operations: Directory and system operations (create_dir, delete_dir, tree, describe)
                   - search_operations: Search operations (search by name or content)
                   - permission_operations: Permission operations (change_permissions)
            
        Returns:
            Tuple of (state, reward, terminated, truncated, info)
        """
       
        # Execute operation based on action type
        operation_result = await self._execute_action(action)
        
        # Calculate reward (simple: +1 for success, -1 for failure)
        success = operation_result.get("success", False)
        reward = 1.0 if success else -1.0
        
        state = dict(
            prompt=await self.file_system.describe(),
        )
        
        # Environment doesn't naturally terminate, so always False
        done = False
        truncated = False
        
        # Create info with operation result
        info = dict(done=done)
        info.update(operation_result)
        
        self.state = state
        self.info = info
        self.done = done
        
        return state, reward, done, truncated, info
    
    async def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action in the file system environment.
        
        Args:
            action: Action dictionary with operation type and parameters
            
        Returns:
            Dictionary with operation result
        """
        try:
            action_type = action.get("type")
            operation = action.get("operation")
            params = action.get("params", {})
            
            if action_type == "file_operations":
                return await self._execute_file_operation(operation, params)
            elif action_type == "directory_operations":
                return await self._execute_directory_operation(operation, params)
            elif action_type == "search_operations":
                return await self._execute_search_operation(operation, params)
            elif action_type == "permission_operations":
                return await self._execute_permission_operation(operation, params)
            else:
                return dict(
                    success=False,
                    action_type=action_type,
                    operation=operation,
                    result=f"Unknown action type: {action_type}",
                )
        
        except Exception as e:
            logger.error(f"| ❌ Error executing action: {e}")
            return dict(
                success=False,
                action_type=action.get("type", "unknown"),
                operation=action.get("operation", "unknown"),
                result=f"Error executing action: {str(e)}",
            )
    
    async def _execute_file_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file operations."""
        try:
            if operation == "read":
                file_path = params.get("file_path")
                start_line = params.get("start_line")
                end_line = params.get("end_line")
                result = await self.file_system.read_file(file_path, start_line=start_line, end_line=end_line)
                
            elif operation == "write":
                file_path = params.get("file_path")
                content = params.get("content")
                mode = params.get("mode", "w")
                result = await self.file_system.write_file(file_path, content, mode=mode)
                
            elif operation == "replace":
                file_path = params.get("file_path")
                old_string = params.get("old_string")
                new_string = params.get("new_string")
                start_line = params.get("start_line")
                end_line = params.get("end_line")
                result = await self.file_system.replace_file_str(
                    file_path, old_string, new_string, start_line=start_line, end_line=end_line
                )
                
            elif operation == "delete":
                file_path = params.get("file_path")
                result = await self.file_system.delete_file(file_path)
                
            elif operation == "copy":
                src_path = params.get("src_path")
                dst_path = params.get("dst_path")
                result = await self.file_system.copy_file(src_path, dst_path)
                
            elif operation == "move":
                src_path = params.get("src_path")
                dst_path = params.get("dst_path")
                result = await self.file_system.move_file(src_path, dst_path)
                
            elif operation == "rename":
                old_path = params.get("old_path")
                new_path = params.get("new_path")
                result = await self.file_system.rename_file(old_path, new_path)
                
            elif operation == "get_info":
                file_path = params.get("file_path")
                include_stats = params.get("include_stats", True)
                result = await self.file_system.get_file_info(file_path, include_stats=include_stats)
                
            else:
                return dict(
                    success=False,
                    action_type="file_operations",
                    operation=operation,
                    result=f"Unknown file operation: {operation}",
                )
            
            success = not result.startswith("Error:")
            result = cleandoc(f"""
            Success: {success}
            Action Type: file_operations
            Operation: {operation}
            Parameters: {params}
            Result: {result}
            """).strip()
            
            return dict(
                success=success,
                action_type="file_operations",
                operation=operation,
                params=params,
                result=result,
            )
            
        except Exception as e:
            logger.error(f"| ❌ Error in file operation '{operation}': {e}")
            return dict(
                success=False,
                action_type="file_operations",
                operation=operation,
                params=params,
                result=f"Error in file operation: {str(e)}",
            )
    
    async def _execute_directory_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute directory operations (directory and system operations)."""
        try:
            if operation == "create_dir":
                dir_path = params.get("dir_path")
                result = await self.file_system.create_directory(dir_path)
                
            elif operation == "delete_dir":
                dir_path = params.get("dir_path")
                result = await self.file_system.delete_directory(dir_path)
                
            elif operation == "tree":
                dir_path = params.get("dir_path")
                max_depth = params.get("max_depth", 3)
                show_hidden = params.get("show_hidden", False)
                exclude_patterns = params.get("exclude_patterns")
                file_types = params.get("file_types")
                result = await self.file_system.tree_structure(
                    dir_path, max_depth=max_depth, show_hidden=show_hidden,
                    exclude_patterns=exclude_patterns, file_types=file_types
                )
                
            elif operation == "describe":
                result = await self.file_system.describe()
                
            else:
                return dict(
                    success=False,
                    action_type="directory_operations",
                    operation=operation,
                    result=f"Unknown dictionary operation: {operation}",
                )
            
            success = not result.startswith("Error:")
            result = cleandoc(f"""
            Success: {success}
            Action Type: directory_operations
            Operation: {operation}
            Parameters: {params}
            Result: {result}
            """).strip()
            
            return dict(
                success=success,
                action_type="directory_operations",
                operation=operation,
                params=params,
                result=result,
            )
            
        except Exception as e:
            logger.error(f"| ❌ Error in dictionary operation '{operation}': {e}")
            return dict(
                success=False,
                action_type="directory_operations",
                operation=operation,
                params=params,
                result=f"Error in dictionary operation: {str(e)}",
            )
    
    async def _execute_search_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search operations."""
        try:
            if operation == "search":
                search_path = params.get("search_path")
                query = params.get("query")
                search_type = params.get("search_type", "name")
                file_types = params.get("file_types")
                case_sensitive = params.get("case_sensitive", False)
                max_results = params.get("max_results", 50)
                result = await self.file_system.search_files(
                    search_path, query=query, search_type=search_type,
                    file_types=file_types, case_sensitive=case_sensitive, max_results=max_results
                )
            else:
                return dict(
                    success=False,
                    action_type="search_operations",
                    operation=operation,
                    result=f"Unknown search operation: {operation}",
                )
            
            success = not result.startswith("Error:")
            result = cleandoc(f"""
            Success: {success}
            Action Type: search_operations
            Operation: {operation}
            Parameters: {params}
            Result: {result}
            """).strip()
            
            return dict(
                success=success,
                action_type="search_operations",
                operation=operation,
                params=params,
                result=result,
            )
            
        except Exception as e:
            logger.error(f"| ❌ Error in search operation '{operation}': {e}")
            return dict(
                success=False,
                action_type="search_operations",
                operation=operation,
                params=params,
                result=f"Error in search operation: {str(e)}",
            )
    
    async def _execute_permission_operation(self, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute permission operations."""
        try:
            if operation == "change_permissions":
                file_path = params.get("file_path")
                permissions = params.get("permissions")
                result = await self.file_system.change_permissions(file_path, permissions)
            else:
                return dict(
                    success=False,
                    action_type="permission_operations",
                    operation=operation,
                    result=f"Unknown permission operation: {operation}",
                )
            
            success = not result.startswith("Error:")
            result = cleandoc(f"""
            Success: {success}
            Action Type: permission_operations
            Operation: {operation}
            Parameters: {params}
            Result: {result}
            """).strip()
            
            return dict(
                success=success,
                action_type="permission_operations",
                operation=operation,
                params=params,
                result=result,
            )
            
        except Exception as e:
            logger.error(f"| ❌ Error in permission operation '{operation}': {e}")
            return dict(
                success=False,
                action_type="permission_operations",
                operation=operation,
                params=params,
                state=f"Error in permission operation: {str(e)}",
            )
    
    
    def close(self):
        """Close the environment and cleanup resources."""
        logger.info("| 🔒 File System Environment closed")
