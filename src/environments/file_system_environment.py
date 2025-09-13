"""File System Environment for AgentWorld - provides file system operations as an environment."""

from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from src.environments.filesystem.file_system import FileSystem
from src.logger import logger
from src.utils import assemble_project_path
from src.environments.protocol.server import ecp
from src.environments.protocol.environment import BaseEnvironment

@ecp.environment(
    name="file_system",
    env_type="file_system",
    description="File system environment for file operations",
    has_vision=False,
    additional_rules={
        "state": "The state of the file system environment [addi].",
    }
)
class FileSystemEnvironment(BaseEnvironment):
    """File System Environment that provides file operations as an environment interface."""
    
    def __init__(
        self,
        base_dir: Union[str, Path],
        create_default_files: bool = True,
        max_file_size: int = 1024 * 1024,  # 1MB
    ):
        """
        Initialize the file system environment.
        
        Args:
            base_dir (Union[str, Path]): Base directory for the file system
            create_default_files (bool): Whether to create default files (todo.md)
            max_file_size (int): Maximum file size in bytes
            allowed_extensions (List[str]): List of allowed file extensions (None for all supported)
        """
        self.base_dir = Path(assemble_project_path(str(base_dir)))
        self.max_file_size = max_file_size
        
        # Initialize file system
        self.file_system = FileSystem(
            base_dir=self.base_dir,
            create_default_files=create_default_files
        )
        
    async def initialize(self) -> None:
        """Initialize the file system environment."""
        logger.info(f"| 🗂️ File System Environment initialized at: {self.base_dir}")
        
    @ecp.action(name = "read", description = "Read a file from the file system.")
    async def _read(self, 
                    file_path: str, 
                    start_line: Optional[int] = None, 
                    end_line: Optional[int] = None) -> str:
        """Read a file from the file system.
        
        Args:
            file_path (str): The absolute path of the file to read.
            start_line (Optional[int]): Start line number for reading a range.
            end_line (Optional[int]): End line number for reading a range.
            
        Returns:
            str: The content of the file.
        """
        return await self.file_system.read_file(file_path, start_line=start_line, end_line=end_line)
    
    @ecp.action(name = "write", 
                description = "Write content to a file.")
    async def _write(self, 
                     file_path: str, 
                     content: str, 
                     mode: str = "w") -> str:
        """Write content to a file.
        
        Args:
            file_path (str): The absolute path of the file to write.
            content (str): The content to write to the file.
            mode (str): Write mode, 'w' for overwrite (default) or 'a' for append.
        
        Returns:
            str: The result of the write operation.
        """
        return await self.file_system.write_file(file_path, content, mode=mode)
    
    @ecp.action(name = "replace", 
                description = "Replace a string in a file.")
    async def _replace(self, 
                       file_path: str, 
                       old_string: str, 
                       new_string: str, 
                       start_line: Optional[int] = None, 
                       end_line: Optional[int] = None) -> str:
        """Replace a string in a file.
        
        Args:
            file_path (str): The absolute path of the file to modify.
            old_string (str): The string to replace.
            new_string (str): The new string to replace with.
            start_line (Optional[int]): Start line number for range replacement.
            end_line (Optional[int]): End line number for range replacement.
        
        Returns:
            str: The result of the replacement.
        """
        return await self.file_system.replace_file_str(file_path, old_string, new_string, start_line=start_line, end_line=end_line)
    
    @ecp.action(name = "delete", 
                description = "Delete a file from the file system.")
    async def _delete(self, file_path: str) -> str:
        """Delete a file from the file system.
        
        Args:
            file_path (str): The absolute path of the file to delete.
            
        Returns:
            str: The result of the deletion.
        """
        return await self.file_system.delete_file(file_path)
    
    @ecp.action(name = "copy", 
                description = "Copy a file from source to destination.")
    async def _copy(self, src_path: str, dst_path: str) -> str:
        """Copy a file from source to destination.
        
        Args:
            src_path (str): The absolute path of the source file.
            dst_path (str): The absolute path of the destination file.
        
        Returns:
            str: The result of the copy operation.
        """
        return await self.file_system.copy_file(src_path, dst_path)
    
    @ecp.action(name = "move",
                description = "Move a file from source to destination.")
    async def _move(self, src_path: str, dst_path: str) -> str:
        """Move a file from source to destination.
        
        Args:
            src_path (str): The absolute path of the source file.
            dst_path (str): The absolute path of the destination file.
        
        Returns:
            str: The result of the move operation.
        """
        return await self.file_system.move_file(src_path, dst_path)
    
    @ecp.action(name = "rename",
                description = "Rename a file or directory.")
    async def _rename(self, old_path: str, new_path: str) -> str:
        """Rename a file or directory.
        
        Args:
            old_path (str): The absolute path of the file/directory to rename.
            new_path (str): The absolute path of the new name.
        
        Returns:
            str: The result of the rename operation.
        """
        return await self.file_system.rename_file(old_path, new_path)
    
    @ecp.action(name = "get_info",
                description = "Get detailed information about a file.")
    async def _get_info(self, file_path: str,
                        include_stats: Optional[bool] = True) -> str:
        """Get detailed information about a file.
        
        Args:
            file_path (str): The absolute path of the file.
            include_stats (Optional[bool]): Whether to include file statistics.
        
        Returns:
            str: The detailed information about the file.
        """
        return await self.file_system.get_file_info(file_path, include_stats=include_stats)
    
    @ecp.action(name = "create_dir",
                description = "Create a directory.")
    async def _create_dir(self, dir_path: str) -> str:
        """Create a directory.
        
        Args:
            dir_path (str): The absolute path of the directory to create.
        
        Returns:
            str: The result of the directory creation.
        """
        return await self.file_system.create_directory(dir_path)
    
    @ecp.action(name = "delete_dir",
                description = "Delete a directory.")
    async def _delete_dir(self, dir_path: str) -> str:
        """Delete a directory.
        
        Args:
            dir_path (str): The absolute path of the directory to delete.
        
        Returns:
            str: The result of the directory deletion.
        """
        return await self.file_system.delete_directory(dir_path)
    
    @ecp.action(name = "tree",
                description = "Show directory tree structure.")
    async def _tree(self, 
                    dir_path: str, 
                    max_depth: Optional[int] = 3, 
                    show_hidden: bool = False, 
                    exclude_patterns: Optional[List[str]] = None, 
                    file_types: Optional[List[str]] = None) -> str:
        """Show directory tree structure.
        
        Args:
            dir_path (str): The absolute path of the directory to show.
            max_depth (Optional[int]): Maximum depth to show.
            show_hidden (Optional[bool]): Whether to show hidden files.
            exclude_patterns (Optional[List[str]]): List of patterns to exclude.
            file_types (Optional[List[str]]): List of file extensions to include.
        
        Returns:
            str: The directory tree structure.
        """
        return await self.file_system.tree_structure(dir_path, max_depth=max_depth, show_hidden=show_hidden, exclude_patterns=exclude_patterns, file_types=file_types)
    
    @ecp.action(name = "describe",
                description = "Describe the file system with directory structure and file information.")
    async def _describe(self) -> str:
        """Describe the file system with directory structure and file information.
        
        Args:
            No parameters required.
        
        Returns:
            str: The file system description with directory structure and file information.
        """
        return await self.file_system.describe()
    
    @ecp.action(name = "search",
                description = "Search for files by name or content.")
    async def _search(self, 
                      search_path: str, 
                      query: str, 
                      search_type: str = "name", 
                      file_types: Optional[List[str]] = None,
                      case_sensitive: Optional[bool] = False, 
                      max_results: Optional[int] = 50) -> str:
        """Search for files by name or content.
        
        Args:
            search_path (str): The absolute path of the directory to search in, or file path for single file search.
            query (str): The search query string.
            search_type (str): Search type, 'name' for filename search or 'content' for content search.
            file_types (Optional[List[str]]): List of file extensions to filter by.
            case_sensitive (Optional[bool]): Whether search is case sensitive.
            max_results (Optional[int]): Maximum number of results to return.
        
        Returns:
            str: The search results.
        """
        return await self.file_system.search_files(search_path, query=query, search_type=search_type, file_types=file_types, case_sensitive=case_sensitive, max_results=max_results)
    
    @ecp.action(name = "change_permissions",
                description = "Change file or directory permissions.")
    async def _change_permissions(self, file_path: str, permissions: str) -> str:
        """Change file or directory permissions.
        
        Args:
            file_path (str): The absolute path of the file or directory.
            permissions (str): The new permissions in octal format (e.g., '755', '644').
        
        Returns:
            str: The result of the permissions change operation.
        """
        return await self.file_system.change_permissions(file_path, permissions)
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the state of the file system environment."""
        state: Dict[str, Any] = {
            "state": await self.file_system.describe()
        }
        return state