"""Project management tool for directory structure, file search, and file listing."""

import asyncio
import os
import re
import json
from typing import Optional, Dict, Any, List
from langchain.tools import BaseTool
from pydantic import Field


_PROJECT_TOOL_DESCRIPTION = """Project management tool that provides comprehensive project structure and file management capabilities.

Available operations:
1. LIST_DIR: List files and directories in a specified path
   - path: directory path to list (absolute or relative)
   - show_hidden: whether to show hidden files (default: False)
   - max_depth: maximum depth for recursive listing (default: 1)
   - file_types: filter by file extensions (optional, e.g., [".py", ".js"])

2. TREE_STRUCTURE: Show directory tree structure
   - root_path: root directory path
   - max_depth: maximum depth for tree (default: 3)
   - show_hidden: whether to show hidden files (default: False)
   - exclude_patterns: patterns to exclude (optional, e.g., ["__pycache__", ".git"])

3. SEARCH_FILES: Search for files by name or content
   - search_path: directory to search in
   - query: search query (filename or content)
   - search_type: "name" for filename search, "content" for content search (default: "name")
   - file_types: filter by file extensions (optional)
   - case_sensitive: whether search is case sensitive (default: False)
   - max_results: maximum number of results (default: 50)

4. GET_FILE_INFO: Get detailed information about a file or directory
   - path: file or directory path
   - include_stats: whether to include file statistics (default: True)

5. FIND_DUPLICATES: Find duplicate files in a directory
   - search_path: directory to search for duplicates
   - file_types: filter by file extensions (optional)
   - check_content: whether to check file content (default: True)

Input format: JSON string with 'operation' and operation-specific parameters.
Example: {"operation": "list_dir", "path": "/path/to/directory", "show_hidden": false}
"""


class ProjectTool(BaseTool):
    """Project management tool for directory structure and file operations."""
    
    name: str = "project"
    description: str = _PROJECT_TOOL_DESCRIPTION
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    async def _arun(self, input_json: str) -> str:
        """Execute project management operation asynchronously based on input JSON."""
        try:
            params = json.loads(input_json)
            operation = params.get("operation", "").upper()
            
            if operation == "LIST_DIR":
                return await self._list_directory(params)
            elif operation == "TREE_STRUCTURE":
                return await self._tree_structure(params)
            elif operation == "SEARCH_FILES":
                return await self._search_files(params)
            elif operation == "GET_FILE_INFO":
                return await self._get_file_info(params)
            elif operation == "FIND_DUPLICATES":
                return await self._find_duplicates(params)
            else:
                return f"Error: Unknown operation '{operation}'. Supported operations: LIST_DIR, TREE_STRUCTURE, SEARCH_FILES, GET_FILE_INFO, FIND_DUPLICATES"
                
        except json.JSONDecodeError:
            return "Error: Invalid JSON input. Please provide valid JSON with 'operation' and parameters."
        except Exception as e:
            return f"Error executing project operation: {str(e)}"
    
    async def _list_directory(self, params: Dict[str, Any]) -> str:
        """List files and directories in a specified path."""
        try:
            path = params.get("path", ".")
            show_hidden = params.get("show_hidden", False)
            max_depth = params.get("max_depth", 1)
            file_types = params.get("file_types", [])
            
            if not os.path.exists(path):
                return f"Error: Path does not exist: {path}"
            
            if not os.path.isdir(path):
                return f"Error: Path is not a directory: {path}"
            
            result = await self._list_directory_recursive(path, show_hidden, max_depth, file_types, 0)
            return result
            
        except Exception as e:
            return f"Error listing directory: {str(e)}"
    
    async def _list_directory_recursive(self, path: str, show_hidden: bool, max_depth: int, 
                                      file_types: List[str], current_depth: int) -> str:
        """Recursively list directory contents."""
        if current_depth > max_depth:
            return ""
        
        try:
            items = []
            for item in os.listdir(path):
                if not show_hidden and item.startswith('.'):
                    continue
                
                item_path = os.path.join(path, item)
                is_dir = os.path.isdir(item_path)
                
                # Filter by file types if specified
                if file_types and not is_dir:
                    file_ext = os.path.splitext(item)[1].lower()
                    if file_ext not in file_types:
                        continue
                
                # Get file size for files
                size_info = ""
                if not is_dir:
                    try:
                        size = os.path.getsize(item_path)
                        size_info = f" ({self._format_size(size)})"
                    except:
                        size_info = " (size unknown)"
                
                prefix = "📁 " if is_dir else "📄 "
                items.append(f"{prefix}{item}{size_info}")
                
                # Recursively list subdirectories
                if is_dir and current_depth < max_depth:
                    sub_path = os.path.join(path, item)
                    sub_items = await self._list_directory_recursive(
                        sub_path, show_hidden, max_depth, file_types, current_depth + 1
                    )
                    if sub_items:
                        for sub_item in sub_items.split('\n'):
                            if sub_item.strip():
                                items.append(f"  {sub_item}")
            
            return f"Directory: {path}\n" + "\n".join(items)
            
        except Exception as e:
            return f"Error reading directory {path}: {str(e)}"
    
    async def _tree_structure(self, params: Dict[str, Any]) -> str:
        """Show directory tree structure."""
        try:
            root_path = params.get("root_path", ".")
            max_depth = params.get("max_depth", 3)
            show_hidden = params.get("show_hidden", False)
            exclude_patterns = params.get("exclude_patterns", [])
            
            if not os.path.exists(root_path):
                return f"Error: Path does not exist: {root_path}"
            
            if not os.path.isdir(root_path):
                return f"Error: Path is not a directory: {root_path}"
            
            tree = await self._build_tree(root_path, "", show_hidden, exclude_patterns, max_depth, 0)
            return f"Project Tree Structure:\n{root_path}\n{tree}"
            
        except Exception as e:
            return f"Error building tree structure: {str(e)}"
    
    async def _build_tree(self, path: str, prefix: str, show_hidden: bool, 
                         exclude_patterns: List[str], max_depth: int, current_depth: int) -> str:
        """Build tree structure recursively."""
        if current_depth >= max_depth:
            return ""
        
        try:
            items = []
            entries = os.listdir(path)
            entries.sort()
            
            for i, entry in enumerate(entries):
                if not show_hidden and entry.startswith('.'):
                    continue
                
                # Check exclude patterns
                if any(re.search(pattern, entry) for pattern in exclude_patterns):
                    continue
                
                entry_path = os.path.join(path, entry)
                is_dir = os.path.isdir(entry_path)
                is_last = i == len(entries) - 1
                
                # Determine tree symbols
                if is_last:
                    tree_symbol = "└── "
                    next_prefix = prefix + "    "
                else:
                    tree_symbol = "├── "
                    next_prefix = prefix + "│   "
                
                # Add file size for files
                size_info = ""
                if not is_dir:
                    try:
                        size = os.path.getsize(entry_path)
                        size_info = f" ({self._format_size(size)})"
                    except:
                        size_info = " (size unknown)"
                
                items.append(f"{prefix}{tree_symbol}{entry}{size_info}")
                
                # Recursively build subtree for directories
                if is_dir:
                    subtree = await self._build_tree(
                        entry_path, next_prefix, show_hidden, exclude_patterns, max_depth, current_depth + 1
                    )
                    if subtree:
                        items.append(subtree)
            
            return "\n".join(items)
            
        except Exception as e:
            return f"Error building tree for {path}: {str(e)}"
    
    async def _search_files(self, params: Dict[str, Any]) -> str:
        """Search for files by name or content."""
        try:
            search_path = params.get("search_path", ".")
            query = params.get("query", "")
            search_type = params.get("search_type", "name")
            file_types = params.get("file_types", [])
            case_sensitive = params.get("case_sensitive", False)
            max_results = params.get("max_results", 50)
            
            if not query:
                return "Error: 'query' is required for SEARCH_FILES operation"
            
            if not os.path.exists(search_path):
                return f"Error: Search path does not exist: {search_path}"
            
            if search_type not in ["name", "content"]:
                return "Error: search_type must be 'name' or 'content'"
            
            results = []
            await self._search_files_recursive(
                search_path, query, search_type, file_types, case_sensitive, results, max_results
            )
            
            if not results:
                return f"No files found matching '{query}' in {search_path}"
            
            result_text = f"Found {len(results)} files matching '{query}' in {search_path}:\n\n"
            for i, result in enumerate(results, 1):
                result_text += f"{i}. {result}\n"
            
            return result_text
            
        except Exception as e:
            return f"Error searching files: {str(e)}"
    
    async def _search_files_recursive(self, path: str, query: str, search_type: str, 
                                    file_types: List[str], case_sensitive: bool, 
                                    results: List[str], max_results: int):
        """Recursively search for files."""
        if len(results) >= max_results:
            return
        
        try:
            for entry in os.listdir(path):
                entry_path = os.path.join(path, entry)
                
                if os.path.isdir(entry_path):
                    await self._search_files_recursive(
                        entry_path, query, search_type, file_types, case_sensitive, results, max_results
                    )
                else:
                    # Check file type filter
                    if file_types:
                        file_ext = os.path.splitext(entry)[1].lower()
                        if file_ext not in file_types:
                            continue
                    
                    if search_type == "name":
                        # Search in filename
                        search_text = entry if case_sensitive else entry.lower()
                        search_query = query if case_sensitive else query.lower()
                        if search_query in search_text:
                            results.append(entry_path)
                    elif search_type == "content":
                        # Search in file content
                        try:
                            with open(entry_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                search_text = content if case_sensitive else content.lower()
                                search_query = query if case_sensitive else query.lower()
                                if search_query in search_text:
                                    results.append(entry_path)
                        except:
                            # Skip files that can't be read as text
                            continue
                            
        except Exception:
            # Skip directories that can't be accessed
            pass
    
    async def _get_file_info(self, params: Dict[str, Any]) -> str:
        """Get detailed information about a file or directory."""
        try:
            path = params.get("path", ".")
            include_stats = params.get("include_stats", True)
            
            if not os.path.exists(path):
                return f"Error: Path does not exist: {path}"
            
            info = []
            info.append(f"Path: {path}")
            info.append(f"Type: {'Directory' if os.path.isdir(path) else 'File'}")
            
            if include_stats:
                try:
                    stat = os.stat(path)
                    info.append(f"Size: {self._format_size(stat.st_size)}")
                    info.append(f"Created: {stat.st_ctime}")
                    info.append(f"Modified: {stat.st_mtime}")
                    info.append(f"Permissions: {oct(stat.st_mode)[-3:]}")
                except Exception as e:
                    info.append(f"Stats error: {str(e)}")
            
            if os.path.isdir(path):
                try:
                    items = os.listdir(path)
                    files = [item for item in items if os.path.isfile(os.path.join(path, item))]
                    dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
                    info.append(f"Contents: {len(files)} files, {len(dirs)} directories")
                except Exception as e:
                    info.append(f"Contents error: {str(e)}")
            
            return "\n".join(info)
            
        except Exception as e:
            return f"Error getting file info: {str(e)}"
    
    async def _find_duplicates(self, params: Dict[str, Any]) -> str:
        """Find duplicate files in a directory."""
        try:
            search_path = params.get("search_path", ".")
            file_types = params.get("file_types", [])
            check_content = params.get("check_content", True)
            
            if not os.path.exists(search_path):
                return f"Error: Search path does not exist: {search_path}"
            
            if not os.path.isdir(search_path):
                return f"Error: Search path is not a directory: {search_path}"
            
            # Collect file information
            files_info = []
            await self._collect_files_info(search_path, file_types, files_info)
            
            # Find duplicates
            duplicates = await self._find_duplicate_files(files_info, check_content)
            
            if not duplicates:
                return f"No duplicate files found in {search_path}"
            
            result = f"Found {len(duplicates)} groups of duplicate files in {search_path}:\n\n"
            for i, group in enumerate(duplicates, 1):
                result += f"Group {i}:\n"
                for file_path in group:
                    result += f"  - {file_path}\n"
                result += "\n"
            
            return result
            
        except Exception as e:
            return f"Error finding duplicates: {str(e)}"
    
    async def _collect_files_info(self, path: str, file_types: List[str], files_info: List[Dict]):
        """Collect information about all files in the directory."""
        try:
            for entry in os.listdir(path):
                entry_path = os.path.join(path, entry)
                
                if os.path.isdir(entry_path):
                    await self._collect_files_info(entry_path, file_types, files_info)
                else:
                    # Check file type filter
                    if file_types:
                        file_ext = os.path.splitext(entry)[1].lower()
                        if file_ext not in file_types:
                            continue
                    
                    try:
                        stat = os.stat(entry_path)
                        files_info.append({
                            'path': entry_path,
                            'size': stat.st_size,
                            'mtime': stat.st_mtime
                        })
                    except:
                        continue
        except Exception:
            pass
    
    async def _find_duplicate_files(self, files_info: List[Dict], check_content: bool) -> List[List[str]]:
        """Find duplicate files based on size and optionally content."""
        # Group by size first
        size_groups = {}
        for file_info in files_info:
            size = file_info['size']
            if size not in size_groups:
                size_groups[size] = []
            size_groups[size].append(file_info)
        
        duplicates = []
        
        for size, files in size_groups.items():
            if len(files) > 1:
                if check_content:
                    # Check content for files with same size
                    content_groups = {}
                    for file_info in files:
                        try:
                            with open(file_info['path'], 'rb') as f:
                                content = f.read()
                                content_hash = hash(content)
                                if content_hash not in content_groups:
                                    content_groups[content_hash] = []
                                content_groups[content_hash].append(file_info['path'])
                        except:
                            continue
                    
                    # Add groups with multiple files
                    for content_hash, file_paths in content_groups.items():
                        if len(file_paths) > 1:
                            duplicates.append(file_paths)
                else:
                    # Just group by size
                    file_paths = [f['path'] for f in files]
                    duplicates.append(file_paths)
        
        return duplicates
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"
    
    def _run(self, input_json: str) -> str:
        """Execute project management operation synchronously."""
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
            "category": "project_management",
            "type": "project"
        }
