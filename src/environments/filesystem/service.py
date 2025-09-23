from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

from src.environments.filesystem.cache import LRUByteCache
from src.environments.filesystem.exceptions import (
    ConflictError,
    InvalidArgumentError,
    NotFoundError,
)
from src.environments.filesystem.handlers import (
    BinaryHandler, CsvHandler, DocxHandler, HandlerRegistry, JsonHandler, 
    MarkdownHandler, PdfHandler, PythonHandler, TextHandler, XlsxHandler
)
from src.environments.filesystem.lock_manager import AsyncLockManager
from src.environments.filesystem.path_policy import PathPolicy
from src.environments.filesystem.storage import LocalAsyncStorage, StorageBackend
from src.environments.filesystem.types import (
    FileReadRequest,
    FileReadResult,
    SearchMatch,
    SearchResult,
    FileWriteRequest, 
    FileWriteResult, 
    FileReplaceRequest, 
    FileReplaceResult,
    FileDeleteRequest, 
    FileDeleteResult,
    FileCopyRequest,
    FileCopyResult,
    FileMoveRequest, 
    FileMoveResult, 
    DirectoryCreateRequest,
    DirectoryCreateResult,
    DirectoryDeleteRequest, 
    DirectoryDeleteResult, 
    FileListRequest, 
    FileListResult,
    FileTreeRequest,
    FileTreeResult, 
    FileSearchRequest,
    FileSearchResult,
    FileStatRequest, 
    FileStatResult, 
    FileStats,
    FileChangePermissionsRequest,
    FileChangePermissionsResult
)


class FileSystemService:
    """Async, sandboxed file system service with handlers, cache, and locks."""

    def __init__(
        self,
        base_dir: Union[str, Path],
        *,
        storage: Optional[StorageBackend] = None,
        cache: Optional[LRUByteCache] = None,
    ) -> None:
        """Initialize the file system service.
        
        Args:
            base_dir: Base directory for file operations
            storage: Storage backend implementation
            cache: Cache implementation for file content
        """
        self._policy = PathPolicy(Path(base_dir) if isinstance(base_dir, str) else base_dir)
        self._storage = storage or LocalAsyncStorage()
        self._cache = cache or LRUByteCache()
        self._locks = AsyncLockManager()

        self._handlers = HandlerRegistry()
        # Register all handlers with priority order (more specific first)
        self._handlers.register(XlsxHandler())
        self._handlers.register(DocxHandler())
        self._handlers.register(PdfHandler())
        self._handlers.register(PythonHandler())
        self._handlers.register(MarkdownHandler())
        self._handlers.register(JsonHandler())
        self._handlers.register(CsvHandler())
        self._handlers.register(BinaryHandler())
        self._handlers.register(TextHandler())  # Fallback handler
        
        # Performance optimization: pre-compile common regex patterns
        self._compiled_patterns: dict[str, re.Pattern] = {}

    # --------------- Helpers ---------------
    def _key(self, relative: Path) -> str:
        """Generate cache key from relative path."""
        return str(relative.as_posix())

    async def _read_raw(self, absolute: Path, relative: Path) -> bytes:
        """Read raw bytes with caching."""
        cache_key = self._key(relative)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        data = await self._storage.read_bytes(absolute)
        self._cache.put(cache_key, data)
        return data

    def _select_handler(self, path: Path):
        """Select appropriate handler for file extension."""
        handler = self._handlers.find_for_extension(path.suffix)
        return handler

    def _compile_pattern(self, pattern: str) -> re.Pattern:
        """Compile regex pattern with caching."""
        if pattern not in self._compiled_patterns:
            self._compiled_patterns[pattern] = re.compile(pattern)
        return self._compiled_patterns[pattern]

    async def _batch_operations(self, operations: List[callable]) -> List[any]:
        """Execute multiple operations concurrently."""
        return await asyncio.gather(*operations, return_exceptions=True)

    # --------------- Public API ---------------
    async def read(self, request: FileReadRequest) -> FileReadResult:
        absolute = self._policy.resolve_relative(request.path)
        relative = self._policy.to_relative(absolute)

        async with self._locks.acquire(self._key(relative)):
            if not await self._storage.exists(absolute):
                raise NotFoundError(f"Path not found: {relative}")
            data = await self._read_raw(absolute, relative)
        handler = self._select_handler(absolute) or TextHandler()
        result = await handler.decode(data, request)
        return result

    async def write(self, request: FileWriteRequest) -> FileWriteResult:
        """Write text content to a file."""
        if request.mode not in {"w", "a"}:
            raise InvalidArgumentError("mode must be 'w' or 'a'")
        
        try:
            absolute = self._policy.resolve_relative(request.path)
            relative = self._policy.to_relative(absolute)
            handler = self._select_handler(absolute) or TextHandler()
            data = await handler.encode(request.content, mode=request.mode, encoding=request.encoding)
            key = self._key(relative)
            
            async with self._locks.acquire(key):
                if request.mode == "a" and await self._storage.exists(absolute):
                    existing = await self._read_raw(absolute, relative)
                    data = existing + data
                await self._storage.write_bytes(absolute, data, overwrite=True)
                self._cache.put(key, data)
            
            return FileWriteResult(
                path=relative,
                bytes_written=len(data),
                success=True,
                message=f"Successfully wrote {len(data)} bytes to {relative}"
            )
        except Exception as e:
            return FileWriteResult(
                path=request.path,
                bytes_written=0,
                success=False,
                message=f"Failed to write file: {e}"
            )

    async def write_bytes(self, path: Path, data: bytes, *, overwrite: bool = True) -> None:
        absolute = self._policy.resolve_relative(path)
        relative = self._policy.to_relative(absolute)
        key = self._key(relative)
        async with self._locks.acquire(key):
            if not overwrite and await self._storage.exists(absolute):
                raise ConflictError(f"Destination exists: {relative}")
            await self._storage.write_bytes(absolute, data, overwrite=True)
            self._cache.put(key, data)

    async def replace(self, request: FileReplaceRequest) -> FileReplaceResult:
        """Replace text in a file."""
        try:
            # Read as text, replace, write back
            read_req = FileReadRequest(
                path=request.path, 
                as_text=True, 
                encoding=request.encoding, 
                start_line=None, 
                end_line=None
            )
            result = await self.read(read_req)
            text = result.content_text or ""
            
            if request.start_line is not None or request.end_line is not None:
                lines = text.splitlines()
                total = len(lines)
                s = (request.start_line - 1) if request.start_line else 0
                e = request.end_line if request.end_line else total
                s = max(0, s)
                e = min(total, e)
                before = "\n".join(lines[:s])
                target = "\n".join(lines[s:e])
                after = "\n".join(lines[e:])
                count = target.count(request.old_string)
                target = target.replace(request.old_string, request.new_string)
                new_text = "\n".join(filter(lambda x: x is not None, [before if before else None, target, after if after else None]))
            else:
                count = text.count(request.old_string)
                new_text = text.replace(request.old_string, request.new_string)
            
            write_req = FileWriteRequest(
                path=request.path,
                content=new_text,
                mode="w",
                encoding=request.encoding
            )
            await self.write_text(write_req)
            
            return FileReplaceResult(
                path=request.path,
                replacements_made=count,
                success=True,
                message=f"Successfully made {count} replacements in {request.path}"
            )
        except Exception as e:
            return FileReplaceResult(
                path=request.path,
                replacements_made=0,
                success=False,
                message=f"Failed to replace text: {e}"
            )

    async def delete(self, request: FileDeleteRequest) -> FileDeleteResult:
        """Remove a file."""
        try:
            absolute = self._policy.resolve_relative(request.path)
            relative = self._policy.to_relative(absolute)
            key = self._key(relative)
            
            async with self._locks.acquire(key):
                if not await self._storage.exists(absolute):
                    raise NotFoundError(f"Path not found: {relative}")
                await self._storage.remove(absolute)
                self._cache.delete(key)
            
            return FileDeleteResult(
                path=relative,
                success=True,
                message=f"Successfully deleted {relative}"
            )
        except Exception as e:
            return FileDeleteResult(
                path=request.path,
                success=False,
                message=f"Failed to delete file: {e}"
            )
            
    async def copy(self, request: FileCopyRequest) -> FileCopyResult:
        """Copy a file."""
        try:
            absolute = self._policy.resolve_relative(request.src_path)
            relative = self._policy.to_relative(absolute)
            async with self._locks.acquire(self._key(relative)):
                if not await self._storage.exists(absolute):
                    raise NotFoundError(f"Path not found: {relative}")
                await self._storage.copy(absolute, request.dst_path)
            
            return FileCopyResult(
                src_path=relative,
                dst_path=request.dst_path,
                bytes_copied=0,
                success=True,
                message=f"Successfully copied {relative} to {request.dst_path}"
            )
        except Exception as e:
            return FileCopyResult(
                src_path=request.src_path,
                dst_path=request.dst_path,
                bytes_copied=0,
                success=False,
                message=f"Failed to copy file: {e}"
            )

    async def rename(self, request: FileMoveRequest) -> FileMoveResult:
        """Rename/move a file."""
        try:
            a_src = self._policy.resolve_relative(request.src_path)
            r_src = self._policy.to_relative(a_src)
            a_dst = self._policy.resolve_relative(request.dst_path)
            r_dst = self._policy.to_relative(a_dst)
            k_src, k_dst = sorted([self._key(r_src), self._key(r_dst)])
            
            async with self._locks.acquire(k_src):
                async with self._locks.acquire(k_dst):
                    if not await self._storage.exists(a_src):
                        raise NotFoundError(f"Path not found: {r_src}")
                    if await self._storage.exists(a_dst) and not request.overwrite:
                        raise ConflictError(f"Destination exists: {r_dst}")
                    await self._storage.rename(a_src, a_dst)
                    # move cache
                    data = self._cache.get(self._key(r_src))
                    self._cache.delete(self._key(r_src))
                    if data is not None:
                        self._cache.put(self._key(r_dst), data)
            
            return FileMoveResult(
                src_path=r_src,
                dst_path=r_dst,
                success=True,
                message=f"Successfully moved {r_src} to {r_dst}"
            )
        except Exception as e:
            return FileMoveResult(
                src_path=request.src_path,
                dst_path=request.dst_path,
                success=False,
                message=f"Failed to move file: {e}"
            )

    async def mkdir(self, request: DirectoryCreateRequest) -> DirectoryCreateResult:
        """Create a directory."""
        try:
            absolute = self._policy.resolve_relative(request.path)
            relative = self._policy.to_relative(absolute)
            
            async with self._locks.acquire(self._key(relative)):
                await self._storage.mkdir(absolute, parents=request.parents)
            
            return DirectoryCreateResult(
                path=relative,
                success=True,
                message=f"Successfully created directory {relative}"
            )
        except Exception as e:
            return DirectoryCreateResult(
                path=request.path,
                success=False,
                message=f"Failed to create directory: {e}"
            )

    async def rmtree(self, request: DirectoryDeleteRequest) -> DirectoryDeleteResult:
        """Remove a directory tree."""
        try:
            absolute = self._policy.resolve_relative(request.path)
            relative = self._policy.to_relative(absolute)
            
            async with self._locks.acquire(self._key(relative)):
                if request.recursive:
                    await self._storage.rmtree(absolute)
                else:
                    # For non-recursive, we need to check if directory is empty
                    entries = await self._storage.listdir(absolute)
                    if entries:
                        raise ConflictError(f"Directory not empty: {relative}")
                    await self._storage.remove(absolute)
            
            return DirectoryDeleteResult(
                path=relative,
                success=True,
                message=f"Successfully deleted directory {relative}"
            )
        except Exception as e:
            return DirectoryDeleteResult(
                path=request.path,
                success=False,
                message=f"Failed to delete directory: {e}"
            )

    async def stat(self, request: FileStatRequest) -> FileStatResult:
        """Get file statistics."""
        try:
            absolute = self._policy.resolve_relative(request.path)
            relative = self._policy.to_relative(absolute)
            
            if not await self._storage.exists(absolute):
                return FileStatResult(
                    path=relative,
                    stats=None,
                    exists=False,
                    success=False,
                    message=f"Path not found: {relative}"
                )
            
            stat_result = await self._storage.stat(absolute)
            
            # Convert os.stat_result to FileStats
            file_stats = FileStats(
                size=stat_result.st_size,
                created=None,  # Not available in os.stat_result
                modified=None,  # Could be converted from st_mtime
                accessed=None,  # Could be converted from st_atime
                permissions=oct(stat_result.st_mode)[-3:],
                is_directory=stat_result.st_mode & 0o170000 == 0o040000,
                is_file=stat_result.st_mode & 0o170000 == 0o100000,
                is_symlink=stat_result.st_mode & 0o170000 == 0o120000
            )
            
            return FileStatResult(
                path=relative,
                stats=file_stats,
                exists=True,
                success=True,
                message=f"Successfully retrieved stats for {relative}"
            )
        except Exception as e:
            return FileStatResult(
                path=request.path,
                stats=None,
                exists=False,
                success=False,
                message=f"Failed to get file stats: {e}"
            )

    async def listdir(self, request: FileListRequest) -> FileListResult:
        """List directory contents."""
        try:
            absolute = self._policy.resolve_relative(request.path)
            relative = self._policy.to_relative(absolute)
            
            entries = await self._storage.listdir(absolute)
            
            files = []
            directories = []
            
            for entry in entries:
                entry_path = absolute / entry
                try:
                    stat_result = await self._storage.stat(entry_path)
                    if stat_result.st_mode & 0o170000 == 0o040000:  # Directory
                        if request.show_hidden or not entry.startswith('.'):
                            directories.append(entry)
                    elif stat_result.st_mode & 0o170000 == 0o100000:  # File
                        if request.show_hidden or not entry.startswith('.'):
                            if request.file_types is None or any(entry.endswith(ext) for ext in request.file_types):
                                files.append(entry)
                except Exception:
                    # If we can't stat, assume it's a file
                    if request.show_hidden or not entry.startswith('.'):
                        if request.file_types is None or any(entry.endswith(ext) for ext in request.file_types):
                            files.append(entry)
            
            return FileListResult(
                path=relative,
                files=files,
                directories=directories,
                total_files=len(files),
                total_directories=len(directories)
            )
        except Exception as e:
            return FileListResult(
                path=request.path,
                files=[],
                directories=[],
                total_files=0,
                total_directories=0
            )

    async def tree(self, request: FileTreeRequest) -> FileTreeResult:
        """Generate directory tree structure with filtering."""
        try:
            absolute = self._policy.resolve_relative(request.path)
            relative = self._policy.to_relative(absolute)
            lines: list[str] = []
            exclude_patterns = request.exclude_patterns or []
            
            # Pre-compile exclude patterns for better performance
            compiled_exclude_patterns = [self._compile_pattern(pattern) for pattern in exclude_patterns]
            
            # Convert file_types to set for O(1) lookup
            file_types_set = set(request.file_types) if request.file_types else None

            async def _walk(current: Path, prefix: str, depth: int) -> None:
                if depth >= request.max_depth:
                    return
                entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                for i, entry in enumerate(entries):
                    name = entry.name
                    if not request.show_hidden and name.startswith('.'):
                        continue
                    if any(compiled_pattern.search(name) for compiled_pattern in compiled_exclude_patterns):
                        continue
                    if file_types_set and entry.is_file() and entry.suffix.lower() not in file_types_set:
                        continue
                    is_last = i == len(entries) - 1
                    connector = '└── ' if is_last else '├── '
                    lines.append(f"{prefix}{connector}{name}")
                    if entry.is_dir():
                        next_prefix = f"{prefix}{'    ' if is_last else '│   '}"
                        await _walk(entry, next_prefix, depth + 1)

            await _walk(absolute, '', 0)
            
            # Count files and directories
            total_files = sum(1 for line in lines if not line.endswith('/'))
            total_directories = sum(1 for line in lines if line.endswith('/'))
            
            return FileTreeResult(
                path=relative,
                tree_lines=lines,
                total_files=total_files,
                total_directories=total_directories
            )
        except Exception as e:
            return FileTreeResult(
                path=request.path,
                tree_lines=[],
                total_files=0,
                total_directories=0
            )

    async def collect_all_files(self, path: Path, *, max_files: int = 100) -> list[Path]:
        """Recursively collect all files under the given path."""
        all_files = []
        
        async def _collect_files(current_path: Path):
            if len(all_files) >= max_files:
                return
            try:
                items = await self.listdir(current_path)
                for item in items:
                    if len(all_files) >= max_files:
                        return
                    item_path = current_path / item
                    # Check if it's a file by using the storage backend
                    absolute_path = self._policy.resolve_relative(item_path)
                    if await self._storage.exists(absolute_path):
                        # Use stat to check if it's a file
                        try:
                            stat = await self._storage.stat(absolute_path)
                            if stat.st_mode & 0o170000 == 0o100000:  # Check if it's a regular file
                                all_files.append(item_path)
                            elif stat.st_mode & 0o170000 == 0o040000:  # Check if it's a directory
                                await _collect_files(item_path)
                        except Exception:
                            # Fallback: assume it's a file if we can't stat it
                            all_files.append(item_path)
            except Exception:
                pass
        
        await _collect_files(path)
        return all_files

    async def search(self, request: FileSearchRequest) -> FileSearchResult:
        """Search for files by name or content with performance optimizations."""
        try:
            absolute = self._policy.resolve_relative(request.path)
            relative = self._policy.to_relative(absolute)
            results: list[SearchResult] = []
            
            # Performance optimizations
            file_types_set = set(request.file_types) if request.file_types else None
            search_query = request.query if request.case_sensitive else request.query.lower()
            max_matches_per_file = 50

            def _match_name(name: str) -> bool:
                """Check if filename matches query."""
                if not request.case_sensitive:
                    return search_query in name.lower()
                return request.query in name

            async def _search_file(file_path: Path) -> Optional[SearchResult]:
                """Search within a single file."""
                if request.by == "name":
                    if _match_name(file_path.name):
                        return SearchResult(path=self._policy.to_relative(file_path), matches=[])
                    return None
                
                # Content search with optimizations
                try:
                    data = await self._storage.read_bytes(file_path)
                    text = data.decode('utf-8', errors='ignore')
                    lines = text.splitlines()
                    matches: list[SearchMatch] = []
                    
                    for idx, line in enumerate(lines, 1):
                        if len(matches) >= max_matches_per_file:
                            break
                        
                        hay = line if request.case_sensitive else line.lower()
                        if search_query in hay:
                            matches.append(SearchMatch(line=idx, text=line))
                    
                    if matches:
                        return SearchResult(path=self._policy.to_relative(file_path), matches=matches)
                except Exception:
                    return None
                return None

            async def _walk(dir_path: Path) -> None:
                """Walk directory tree and search files."""
                nonlocal results
                try:
                    entries = list(dir_path.iterdir())
                    # Process directories first, then files for better performance
                    dirs = [e for e in entries if e.is_dir()]
                    files = [e for e in entries if e.is_file()]
                    
                    # Process directories concurrently
                    if dirs:
                        tasks = [_walk(d) for d in dirs]
                        await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process files
                    for entry in files:
                        if len(results) >= request.max_results:
                            return
                        if file_types_set and entry.suffix.lower() not in file_types_set:
                            continue
                        res = await _search_file(entry)
                        if res:
                            results.append(res)
                            if len(results) >= request.max_results:
                                return
                except Exception:
                    pass  # Skip directories we can't access

            if absolute.is_file():
                single = await _search_file(absolute)
                results = [single] if single else []
            else:
                await _walk(absolute)
            
            return FileSearchResult(
                query=request.query,
                search_by=request.by,
                results=results,
                total_found=len(results)
            )
        except Exception as e:
            return FileSearchResult(
                query=request.query,
                search_by=request.by,
                results=[],
                total_found=0
            )

    async def change_permissions(self, request: FileChangePermissionsRequest) -> FileChangePermissionsResult:
        """Change file or directory permissions."""
        try:
            absolute = self._policy.resolve_relative(request.path)
            relative = self._policy.to_relative(absolute)
            
            async with self._locks.acquire(self._key(relative)):
                await self._storage.chmod(absolute, request.permissions)

            return FileChangePermissionsResult(
                path=relative,
                success=True,
                message=f"Successfully changed permissions for {relative}"
            )
        except Exception as e:
            return FileChangePermissionsResult(
                path=request.path,
                success=False,
                message=f"Failed to change permissions: {e}"
            )
    async def describe(self) -> str:
        """Describe the file system."""
        return "The file system is a file system that provides file operations as an environment interface."