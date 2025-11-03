"""File System Environment for AgentWorld - provides file system operations as an environment."""

from pathlib import Path
from typing import List, Optional, Any, Dict, Type
from pydantic import BaseModel, Field, ConfigDict

from src.environments.filesystem.service import FileSystemService
from src.environments.filesystem.types import (
    FileReadRequest, 
    FileWriteRequest,
    FileReplaceRequest, 
    FileDeleteRequest,
    FileCopyRequest,
    FileMoveRequest,
    DirectoryCreateRequest, 
    DirectoryDeleteRequest,
    FileListRequest, 
    FileTreeRequest, 
    FileSearchRequest, 
    FileStatRequest,
    FileChangePermissionsRequest
)
from src.logger import logger
from src.utils import assemble_project_path
from src.environments.protocol.server import ecp
from src.environments.protocol.environment import BaseEnvironment
from src.utils import dedent

@ecp.environment()
class FileSystemEnvironment(BaseEnvironment):
    """File System Environment hat provides file operations as an environment interface."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="file_system", description="The name of the file system environment.")
    type: str = Field(default="File System", description="The type of the file system environment.")
    description: str = Field(default="File system environment for file operations", description="The description of the file system environment.")
    args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the file system environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the file system environment.",
        }
    }, description="The metadata of the file system environment.")
    
    def __init__(
        self,
        base_dir: str = None,
        max_file_size: int = 1024 * 1024,  # 1MB
        **kwargs
    ):
        """
        Initialize the file system environment.
        
        Args:
            base_dir (str): Base directory for the file system
            max_file_size (int): Maximum file size in bytes
        """
        super().__init__(**kwargs)
        self.base_dir = assemble_project_path(base_dir)
        self.max_file_size = max_file_size
        
        # Initialize file system
        self.file_system_service = FileSystemService(
            base_dir=self.base_dir
        )
        
    async def initialize(self) -> None:
        """Initialize the file system environment."""
        logger.info(f"| 🗂️ File System Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the file system environment."""
        logger.info("| 🧹 File System Environment cleanup completed")
        
    @ecp.action(name = "read", 
                type = "File System", 
                description = "Read a file from the file system.")
    async def read(self, 
                    file_path: str, 
                    start_line: Optional[int] = None, 
                    end_line: Optional[int] = None) -> Dict[str, Any]:
        """Read a file from the file system.
        
        Args:
            file_path (str): The absolute path of the file to read.
            start_line (Optional[int]): Start line number for reading a range.
            end_line (Optional[int]): End line number for reading a range.
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileReadRequest(
                path=Path(file_path),
                start_line=start_line,
                end_line=end_line
            )
            result = await self.file_system_service.read(request)
            
            extra = result.extra.copy() if result.extra else {}
            
            if result.success:
                if "content_text" in extra:
                    message = extra["content_text"]
                elif "content_bytes_length" in extra:
                    message = f"File read successfully. Content length: {extra['content_bytes_length']} bytes"
                else:
                    message = result.message
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to read file: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path}
            }
    
    @ecp.action(name = "write", 
                type = "File System", 
                description = "Write content to a file.")
    async def write(self, 
                     file_path: str, 
                     content: str, 
                     mode: str = "w") -> Dict[str, Any]:
        """Write content to a file.
        
        Args:
            file_path (str): The absolute path of the file to write.
            content (str): The content to write to the file.
            mode (str): Write mode, 'w' for overwrite (default) or 'a' for append.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileWriteRequest(
                path=Path(file_path),
                content=content,
                mode=mode
            )
            result = await self.file_system_service.write(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["file_path"] = file_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to write file: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path}
            }
    
    @ecp.action(name = "replace", 
                type = "File System", 
                description = "Replace a string in a file.")
    async def replace(self, 
                       file_path: str, 
                       old_string: str, 
                       new_string: str, 
                       start_line: Optional[int] = None, 
                       end_line: Optional[int] = None) -> Dict[str, Any]:
        """Replace a string in a file.
        
        Args:
            file_path (str): The absolute path of the file to modify.
            old_string (str): The string to replace.
            new_string (str): The new string to replace with.
            start_line (Optional[int]): Start line number for range replacement.
            end_line (Optional[int]): End line number for range replacement.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileReplaceRequest(
                path=Path(file_path),
                old_string=old_string,
                new_string=new_string,
                start_line=start_line,
                end_line=end_line
            )
            result = await self.file_system_service.replace(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["file_path"] = file_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to replace text: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path}
            }
    
    @ecp.action(name = "delete", 
                type = "File System", 
                description = "Delete a file from the file system.")
    async def delete(self, file_path: str) -> Dict[str, Any]:
        """Delete a file from the file system.
        
        Args:
            file_path (str): The absolute path of the file to delete.
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileDeleteRequest(path=Path(file_path))
            result = await self.file_system_service.delete(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["file_path"] = file_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete file: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path}
            }
    
    @ecp.action(name = "copy", 
                type = "File System", 
                description = "Copy a file from source to destination.")
    async def copy(self, src_path: str, dst_path: str) -> Dict[str, Any]:
        """Copy a file from source to destination.
        
        Args:
            src_path (str): The absolute path of the source file.
            dst_path (str): The absolute path of the destination file.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileCopyRequest(src_path=Path(src_path), dst_path=Path(dst_path))
            result = await self.file_system_service.copy(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["src_path"] = src_path
            extra["dst_path"] = dst_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to copy file: {str(e)}",
                "extra": {"error": str(e), "src_path": src_path, "dst_path": dst_path}
            }
    
    @ecp.action(name = "move",
                type = "File System", 
                description = "Move a file from source to destination.")
    async def move(self, src_path: str, dst_path: str) -> Dict[str, Any]:
        """Move a file from source to destination.
        
        Args:
            src_path (str): The absolute path of the source file.
            dst_path (str): The absolute path of the destination file.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileMoveRequest(
                src_path=Path(src_path),
                dst_path=Path(dst_path)
            )
            result = await self.file_system_service.rename(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["src_path"] = src_path
            extra["dst_path"] = dst_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to move file: {str(e)}",
                "extra": {"error": str(e), "src_path": src_path, "dst_path": dst_path}
            }
    
    @ecp.action(name = "rename",
                type = "File System", 
                description = "Rename a file or directory.")
    async def rename(self, old_path: str, new_path: str) -> Dict[str, Any]:
        """Rename a file or directory.
        
        Args:
            old_path (str): The absolute path of the file/directory to rename.
            new_path (str): The absolute path of the new name.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileMoveRequest(
                src_path=Path(old_path),
                dst_path=Path(new_path)
            )
            result = await self.file_system_service.rename(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["old_path"] = old_path
            extra["new_path"] = new_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to rename: {str(e)}",
                "extra": {"error": str(e), "old_path": old_path, "new_path": new_path}
            }
    
    @ecp.action(name = "get_info",
                type = "File System", 
                description = "Get detailed information about a file.")
    async def get_info(self, file_path: str,
                        include_stats: Optional[bool] = True) -> Dict[str, Any]:
        """Get detailed information about a file.
        
        Args:
            file_path (str): The absolute path of the file.
            include_stats (Optional[bool]): Whether to include file statistics.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileStatRequest(path=Path(file_path))
            result = await self.file_system_service.stat(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["file_path"] = file_path
            extra["include_stats"] = include_stats
            
            if result.success and "stats" in extra:
                stats = extra["stats"]
                info = f"File: {file_path}\n"
                info += f"Size: {stats.get('size', 0)} bytes\n"
                info += f"Type: {'Directory' if stats.get('is_directory', False) else 'File'}\n"
                info += f"Permissions: {stats.get('permissions', 'Unknown')}\n"
                if include_stats:
                    info += f"Is Directory: {stats.get('is_directory', False)}\n"
                    info += f"Is File: {stats.get('is_file', False)}\n"
                    info += f"Is Symlink: {stats.get('is_symlink', False)}\n"
                message = info
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get file info: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path}
            }
    
    @ecp.action(name = "create_dir",
                type = "File System", 
                description = "Create a directory.")
    async def create_dir(self, dir_path: str) -> Dict[str, Any]:
        """Create a directory.
        
        Args:
            dir_path (str): The absolute path of the directory to create.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = DirectoryCreateRequest(path=Path(dir_path))
            result = await self.file_system_service.mkdir(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["dir_path"] = dir_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create directory: {str(e)}",
                "extra": {"error": str(e), "dir_path": dir_path}
            }
    
    @ecp.action(name = "delete_dir",
                type = "File System", 
                description = "Delete a directory.")
    async def delete_dir(self, dir_path: str) -> Dict[str, Any]:
        """Delete a directory.
        
        Args:
            dir_path (str): The absolute path of the directory to delete.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = DirectoryDeleteRequest(path=Path(dir_path), recursive=True)
            result = await self.file_system_service.rmtree(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["dir_path"] = dir_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete directory: {str(e)}",
                "extra": {"error": str(e), "dir_path": dir_path}
            }
    
    @ecp.action(name = "listdir",
                type = "File System", 
                description = "List directory contents.")
    async def listdir(self, 
                       dir_path: str, 
                       show_hidden: bool = False, 
                       file_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """List directory contents.
        
        Args:
            dir_path (str): The absolute path of the directory to list.
            show_hidden (bool): Whether to show hidden files and directories.
            file_types (Optional[List[str]]): List of file extensions to filter by.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileListRequest(
                path=Path(dir_path),
                show_hidden=show_hidden,
                file_types=file_types
            )
            result = await self.file_system_service.listdir(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["dir_path"] = dir_path
            
            if result.success:
                if extra.get("files") or extra.get("directories"):
                    listing = f"Contents of {dir_path}:\n"
                    listing += f"Total: {extra.get('total_files', 0)} files, {extra.get('total_directories', 0)} directories\n\n"
                    
                    if extra.get("directories"):
                        listing += "Directories:\n"
                        for directory in extra["directories"]:
                            listing += f"  📁 {directory}/\n"
                        listing += "\n"
                    
                    if extra.get("files"):
                        listing += "Files:\n"
                        for file in extra["files"]:
                            listing += f"  📄 {file}\n"
                    
                    message = listing
                else:
                    message = f"Directory {dir_path} is empty"
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to list directory: {str(e)}",
                "extra": {"error": str(e), "dir_path": dir_path}
            }
    
    @ecp.action(name = "tree",
                type = "File System", 
                description = "Show directory tree structure.")
    async def tree(self, 
                    dir_path: str, 
                    max_depth: Optional[int] = 3, 
                    show_hidden: bool = False, 
                    exclude_patterns: Optional[List[str]] = None, 
                    file_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Show directory tree structure.
        
        Args:
            dir_path (str): The absolute path of the directory to show.
            max_depth (Optional[int]): Maximum depth to show.
            show_hidden (Optional[bool]): Whether to show hidden files.
            exclude_patterns (Optional[List[str]]): List of patterns to exclude.
            file_types (Optional[List[str]]): List of file extensions to include.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileTreeRequest(
                path=Path(dir_path),
                max_depth=max_depth,
                show_hidden=show_hidden,
                exclude_patterns=exclude_patterns,
                file_types=file_types
            )
            result = await self.file_system_service.tree(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["dir_path"] = dir_path
            
            if result.success:
                if extra.get("tree_lines"):
                    tree_str = f"Directory tree for {dir_path}:\n"
                    tree_str += "\n".join(extra["tree_lines"])
                    tree_str += f"\n\nTotal: {extra.get('total_files', 0)} files, {extra.get('total_directories', 0)} directories"
                    message = tree_str
                else:
                    message = f"No tree structure found for {dir_path}"
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to generate tree: {str(e)}",
                "extra": {"error": str(e), "dir_path": dir_path}
            }
    
    @ecp.action(name = "describe",
                type = "File System", 
                description = "Describe the file system with directory structure and file information.")
    async def describe(self) -> Dict[str, Any]:
        """Describe the file system with directory structure and file information.
        
        Args:
            No parameters required.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # Use tree to describe the file system
            request = FileTreeRequest(
                path=Path("."),
                max_depth=3,
                show_hidden=False
            )
            result = await self.file_system_service.tree(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["base_dir"] = str(self.base_dir)
            
            if result.success:
                description = f"File System Environment at: {self.base_dir}\n"
                description += f"Total: {extra.get('total_files', 0)} files, {extra.get('total_directories', 0)} directories\n\n"
                
                if extra.get("tree_lines"):
                    description += "Directory Structure:\n"
                    description += "\n".join(extra["tree_lines"])
                else:
                    description += "No files or directories found."
                
                message = description
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to describe file system: {str(e)}",
                "extra": {"error": str(e), "base_dir": str(self.base_dir)}
            }
    
    @ecp.action(name = "search",
                type = "File System", 
                description = "Search for files by name or content.")
    async def search(self, 
                      search_path: str, 
                      query: str, 
                      search_type: str = "name", 
                      file_types: Optional[List[str]] = None,
                      case_sensitive: Optional[bool] = False, 
                      max_results: Optional[int] = 50) -> Dict[str, Any]:
        """Search for files by name or content.
        
        Args:
            search_path (str): The absolute path of the directory to search in, or file path for single file search.
            query (str): The search query string.
            search_type (str): Search type, 'name' for filename search or 'content' for content search.
            file_types (Optional[List[str]]): List of file extensions to filter by.
            case_sensitive (Optional[bool]): Whether search is case sensitive.
            max_results (Optional[int]): Maximum number of results to return.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileSearchRequest(
                path=Path(search_path),
                query=query,
                by=search_type,
                file_types=file_types,
                case_sensitive=case_sensitive,
                max_results=max_results
            )
            result = await self.file_system_service.search(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["search_path"] = search_path
            extra["query"] = query
            
            if result.success:
                if extra.get("results"):
                    search_str = f"Search results for '{query}' in {search_path}:\n"
                    search_str += f"Found {extra.get('total_found', 0)} results\n\n"
                    
                    results = extra["results"]
                    for i, search_result in enumerate(results, 1):
                        search_str += f"{i}. {search_result.get('path', 'Unknown')}\n"
                        if search_result.get("matches"):
                            for match in search_result["matches"][:5]:  # Show first 5 matches
                                search_str += f"   Line {match.get('line', 0)}: {match.get('text', '')[:100]}...\n"
                        search_str += "\n"
                    
                    message = search_str
                else:
                    message = f"No results found for '{query}' in {search_path}"
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to search files: {str(e)}",
                "extra": {"error": str(e), "search_path": search_path, "query": query}
            }
    
    @ecp.action(name = "change_permissions",
                type = "File System", 
                description = "Change file or directory permissions.")
    async def change_permissions(self, file_path: str, permissions: str) -> Dict[str, Any]:
        """Change file or directory permissions.
        
        Args:
            file_path (str): The absolute path of the file or directory.
            permissions (str): The new permissions in octal format (e.g., '755', '644').
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = FileChangePermissionsRequest(path=Path(file_path), permissions=permissions)
            result = await self.file_system_service.change_permissions(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["file_path"] = file_path
            extra["permissions"] = permissions
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to change permissions: {str(e)}",
                "extra": {"error": str(e), "file_path": file_path, "permissions": permissions}
            }
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the state of the file system environment."""
        try:
            state = await self.describe()
            state = dedent(f"""
                <info>
                {state["message"]}
                </info>
            """)
            extra = state["extra"]
            return {
                "state": state,
                "extra": extra
            }
        except Exception as e:
            logger.error(f"Failed to get file system state: {e}")
            return {
                "state": f"Failed to get file system state: {str(e)}",
                "extra": {
                    "error": str(e)
                }
            }