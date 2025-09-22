"""File System Environment for AgentWorld - provides file system operations as an environment."""

from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from src.environments.filesystem.file_system import FileSystem
from src.environments.filesystem.types import (
    FileReadRequest, 
    FileWriteRequest,
    FileReplaceRequest, 
    FileDeleteRequest,
    FileMoveRequest,
    DirectoryCreateRequest, 
    DirectoryDeleteRequest,
    FileListRequest, 
    FileTreeRequest, 
    FileSearchRequest, 
    FileStatRequest
)
from src.logger import logger
from src.utils import assemble_project_path
from src.environments.protocol.server import ecp
from src.environments.protocol.environment import BaseEnvironment

@ecp.environment(
    name="file_system",
    type="File System",
    description="File system environment for file operations",
    has_vision=False,
    additional_rules={
        "state": "The state of the file system environment.",
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
        
    @ecp.action(name = "read", 
                type = "File System", 
                description = "Read a file from the file system.")
    async def read(self, 
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
        request = FileReadRequest(
            path=Path(file_path),
            start_line=start_line,
            end_line=end_line
        )
        result = await self.file_system.read(request)
        
        if result.content_text:
            return result.content_text
        elif result.content_bytes:
            return result.content_bytes.decode('utf-8', errors='ignore')
        else:
            return f"File read result: {result.message}"
    
    @ecp.action(name = "write", 
                type = "File System", 
                description = "Write content to a file.")
    async def write(self, 
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
        request = FileWriteRequest(
            path=Path(file_path),
            content=content,
            mode=mode
        )
        result = await self.file_system.write_text(request)
        return result.message
    
    @ecp.action(name = "replace", 
                type = "File System", 
                description = "Replace a string in a file.")
    async def replace(self, 
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
        request = FileReplaceRequest(
            path=Path(file_path),
            old_string=old_string,
            new_string=new_string,
            start_line=start_line,
            end_line=end_line
        )
        result = await self.file_system.replace(request)
        return result.message
    
    @ecp.action(name = "delete", 
                type = "File System", 
                description = "Delete a file from the file system.")
    async def delete(self, file_path: str) -> str:
        """Delete a file from the file system.
        
        Args:
            file_path (str): The absolute path of the file to delete.
            
        Returns:
            str: The result of the deletion.
        """
        request = FileDeleteRequest(path=Path(file_path))
        result = await self.file_system.remove(request)
        return result.message
    
    @ecp.action(name = "copy", 
                type = "File System", 
                description = "Copy a file from source to destination.")
    async def copy(self, src_path: str, dst_path: str) -> str:
        """Copy a file from source to destination.
        
        Args:
            src_path (str): The absolute path of the source file.
            dst_path (str): The absolute path of the destination file.
        
        Returns:
            str: The result of the copy operation.
        """
        # For copy operation, we need to read the source file and write to destination
        read_request = FileReadRequest(path=Path(src_path))
        read_result = await self.file_system.read(read_request)
        
        if not read_result.content_bytes:
            return f"Failed to read source file: {src_path}"
        
        write_request = FileWriteRequest(
            path=Path(dst_path),
            content=read_result.content_bytes.decode('utf-8', errors='ignore'),
            mode="w"
        )
        write_result = await self.file_system.write_text(write_request)
        return write_result.message
    
    @ecp.action(name = "move",
                type = "File System", 
                description = "Move a file from source to destination.")
    async def move(self, src_path: str, dst_path: str) -> str:
        """Move a file from source to destination.
        
        Args:
            src_path (str): The absolute path of the source file.
            dst_path (str): The absolute path of the destination file.
        
        Returns:
            str: The result of the move operation.
        """
        request = FileMoveRequest(
            src_path=Path(src_path),
            dst_path=Path(dst_path)
        )
        result = await self.file_system.rename(request)
        return result.message
    
    @ecp.action(name = "rename",
                type = "File System", 
                description = "Rename a file or directory.")
    async def rename(self, old_path: str, new_path: str) -> str:
        """Rename a file or directory.
        
        Args:
            old_path (str): The absolute path of the file/directory to rename.
            new_path (str): The absolute path of the new name.
        
        Returns:
            str: The result of the rename operation.
        """
        request = FileMoveRequest(
            src_path=Path(old_path),
            dst_path=Path(new_path)
        )
        result = await self.file_system.rename(request)
        return result.message
    
    @ecp.action(name = "get_info",
                type = "File System", 
                description = "Get detailed information about a file.")
    async def get_info(self, file_path: str,
                        include_stats: Optional[bool] = True) -> str:
        """Get detailed information about a file.
        
        Args:
            file_path (str): The absolute path of the file.
            include_stats (Optional[bool]): Whether to include file statistics.
        
        Returns:
            str: The detailed information about the file.
        """
        request = FileStatRequest(path=Path(file_path))
        result = await self.file_system.stat(request)
        
        if result.success and result.stats:
            stats = result.stats
            info = f"File: {file_path}\n"
            info += f"Size: {stats.size} bytes\n"
            info += f"Type: {'Directory' if stats.is_directory else 'File'}\n"
            info += f"Permissions: {stats.permissions}\n"
            if include_stats:
                info += f"Is Directory: {stats.is_directory}\n"
                info += f"Is File: {stats.is_file}\n"
                info += f"Is Symlink: {stats.is_symlink}\n"
            return info
        else:
            return result.message
    
    @ecp.action(name = "create_dir",
                type = "File System", 
                description = "Create a directory.")
    async def create_dir(self, dir_path: str) -> str:
        """Create a directory.
        
        Args:
            dir_path (str): The absolute path of the directory to create.
        
        Returns:
            str: The result of the directory creation.
        """
        request = DirectoryCreateRequest(path=Path(dir_path))
        result = await self.file_system.mkdir(request)
        return result.message
    
    @ecp.action(name = "delete_dir",
                type = "File System", 
                description = "Delete a directory.")
    async def delete_dir(self, dir_path: str) -> str:
        """Delete a directory.
        
        Args:
            dir_path (str): The absolute path of the directory to delete.
        
        Returns:
            str: The result of the directory deletion.
        """
        request = DirectoryDeleteRequest(path=Path(dir_path), recursive=True)
        result = await self.file_system.rmtree(request)
        return result.message
    
    @ecp.action(name = "listdir",
                type = "File System", 
                description = "List directory contents.")
    async def listdir(self, 
                       dir_path: str, 
                       show_hidden: bool = False, 
                       file_types: Optional[List[str]] = None) -> str:
        """List directory contents.
        
        Args:
            dir_path (str): The absolute path of the directory to list.
            show_hidden (bool): Whether to show hidden files and directories.
            file_types (Optional[List[str]]): List of file extensions to filter by.
        
        Returns:
            str: The directory contents listing.
        """
        request = FileListRequest(
            path=Path(dir_path),
            show_hidden=show_hidden,
            file_types=file_types
        )
        result = await self.file_system.listdir(request)
        
        if result.files or result.directories:
            listing = f"Contents of {dir_path}:\n"
            listing += f"Total: {result.total_files} files, {result.total_directories} directories\n\n"
            
            if result.directories:
                listing += "Directories:\n"
                for directory in result.directories:
                    listing += f"  📁 {directory}/\n"
                listing += "\n"
            
            if result.files:
                listing += "Files:\n"
                for file in result.files:
                    listing += f"  📄 {file}\n"
            
            return listing
        else:
            return f"Directory {dir_path} is empty"
    
    @ecp.action(name = "tree",
                type = "File System", 
                description = "Show directory tree structure.")
    async def tree(self, 
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
        request = FileTreeRequest(
            path=Path(dir_path),
            max_depth=max_depth,
            show_hidden=show_hidden,
            exclude_patterns=exclude_patterns,
            file_types=file_types
        )
        result = await self.file_system.tree(request)
        
        if result.tree_lines:
            tree_str = f"Directory tree for {dir_path}:\n"
            tree_str += "\n".join(result.tree_lines)
            tree_str += f"\n\nTotal: {result.total_files} files, {result.total_directories} directories"
            return tree_str
        else:
            return f"No tree structure found for {dir_path}"
    
    @ecp.action(name = "describe",
                type = "File System", 
                description = "Describe the file system with directory structure and file information.")
    async def describe(self) -> str:
        """Describe the file system with directory structure and file information.
        
        Args:
            No parameters required.
        
        Returns:
            str: The file system description with directory structure and file information.
        """
        # Use tree to describe the file system
        request = FileTreeRequest(
            path=Path("."),
            max_depth=3,
            show_hidden=False
        )
        result = await self.file_system.tree(request)
        
        description = f"File System Environment at: {self.base_dir}\n"
        description += f"Total: {result.total_files} files, {result.total_directories} directories\n\n"
        
        if result.tree_lines:
            description += "Directory Structure:\n"
            description += "\n".join(result.tree_lines)
        else:
            description += "No files or directories found."
        
        return description
    
    @ecp.action(name = "search",
                type = "File System", 
                description = "Search for files by name or content.")
    async def search(self, 
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
        request = FileSearchRequest(
            path=Path(search_path),
            query=query,
            by=search_type,
            file_types=file_types,
            case_sensitive=case_sensitive,
            max_results=max_results
        )
        result = await self.file_system.search(request)
        
        if result.results:
            search_str = f"Search results for '{query}' in {search_path}:\n"
            search_str += f"Found {result.total_found} results\n\n"
            
            for i, search_result in enumerate(result.results, 1):
                search_str += f"{i}. {search_result.path}\n"
                if search_result.matches:
                    for match in search_result.matches[:5]:  # Show first 5 matches
                        search_str += f"   Line {match.line}: {match.text[:100]}...\n"
                search_str += "\n"
            
            return search_str
        else:
            return f"No results found for '{query}' in {search_path}"
    
    @ecp.action(name = "change_permissions",
                type = "File System", 
                description = "Change file or directory permissions.")
    async def change_permissions(self, file_path: str, permissions: str) -> str:
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