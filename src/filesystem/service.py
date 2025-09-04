from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

from src.filesystem.cache import LRUByteCache
from src.filesystem.exceptions import (
    ConflictError,
    InvalidArgumentError,
    NotFoundError,
)
from src.filesystem.handlers import (
    BinaryHandler, CsvHandler, HandlerRegistry, JsonHandler, 
    MarkdownHandler, PythonHandler, TextHandler
)
from src.filesystem.lock_manager import AsyncLockManager
from src.filesystem.path_policy import PathPolicy
from src.filesystem.storage import LocalAsyncStorage, StorageBackend
from src.filesystem.types import FileReadRequest, FileReadResult, SearchMatch, SearchResult


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

    async def write_text(self, path: Path, text: str, *, mode: str = "w", encoding: str = "utf-8") -> None:
        if mode not in {"w", "a"}:
            raise InvalidArgumentError("mode must be 'w' or 'a'")
        absolute = self._policy.resolve_relative(path)
        relative = self._policy.to_relative(absolute)
        handler = self._select_handler(absolute) or TextHandler()
        data = await handler.encode(text, mode=mode, encoding=encoding)
        key = self._key(relative)
        async with self._locks.acquire(key):
            if mode == "a" and await self._storage.exists(absolute):
                existing = await self._read_raw(absolute, relative)
                data = existing + data
            await self._storage.write_bytes(absolute, data, overwrite=True)
            self._cache.put(key, data)

    async def write_bytes(self, path: Path, data: bytes, *, overwrite: bool = True) -> None:
        absolute = self._policy.resolve_relative(path)
        relative = self._policy.to_relative(absolute)
        key = self._key(relative)
        async with self._locks.acquire(key):
            if not overwrite and await self._storage.exists(absolute):
                raise ConflictError(f"Destination exists: {relative}")
            await self._storage.write_bytes(absolute, data, overwrite=True)
            self._cache.put(key, data)

    async def replace(self, path: Path, old: str, new: str, *, start_line: Optional[int] = None, end_line: Optional[int] = None, encoding: str = "utf-8") -> int:
        # Read as text, replace, write back
        req = FileReadRequest(path=path, as_text=True, encoding=encoding, start_line=None, end_line=None)
        result = await self.read(req)
        text = result.content_text or ""
        if start_line is not None or end_line is not None:
            lines = text.splitlines()
            total = len(lines)
            s = (start_line - 1) if start_line else 0
            e = end_line if end_line else total
            s = max(0, s)
            e = min(total, e)
            before = "\n".join(lines[:s])
            target = "\n".join(lines[s:e])
            after = "\n".join(lines[e:])
            count = target.count(old)
            target = target.replace(old, new)
            new_text = "\n".join(filter(lambda x: x is not None, [before if before else None, target, after if after else None]))
        else:
            count = text.count(old)
            new_text = text.replace(old, new)
        await self.write_text(path, new_text, mode="w", encoding=encoding)
        return count

    async def remove(self, path: Path) -> None:
        absolute = self._policy.resolve_relative(path)
        relative = self._policy.to_relative(absolute)
        key = self._key(relative)
        async with self._locks.acquire(key):
            if not await self._storage.exists(absolute):
                raise NotFoundError(f"Path not found: {relative}")
            try:
                await self._storage.remove(absolute)
            finally:
                self._cache.delete(key)

    async def rename(self, src: Path, dst: Path) -> None:
        a_src = self._policy.resolve_relative(src)
        r_src = self._policy.to_relative(a_src)
        a_dst = self._policy.resolve_relative(dst)
        r_dst = self._policy.to_relative(a_dst)
        k_src, k_dst = sorted([self._key(r_src), self._key(r_dst)])
        async with self._locks.acquire(k_src):
            async with self._locks.acquire(k_dst):
                if not await self._storage.exists(a_src):
                    raise NotFoundError(f"Path not found: {r_src}")
                if await self._storage.exists(a_dst):
                    raise ConflictError(f"Destination exists: {r_dst}")
                await self._storage.rename(a_src, a_dst)
                # move cache
                data = self._cache.get(self._key(r_src))
                self._cache.delete(self._key(r_src))
                if data is not None:
                    self._cache.put(self._key(r_dst), data)

    async def mkdir(self, path: Path) -> None:
        absolute = self._policy.resolve_relative(path)
        relative = self._policy.to_relative(absolute)
        async with self._locks.acquire(self._key(relative)):
            await self._storage.mkdir(absolute, parents=True)

    async def rmtree(self, path: Path) -> None:
        absolute = self._policy.resolve_relative(path)
        relative = self._policy.to_relative(absolute)
        async with self._locks.acquire(self._key(relative)):
            await self._storage.rmtree(absolute)

    async def stat(self, path: Path) -> os.stat_result:
        absolute = self._policy.resolve_relative(path)
        return await self._storage.stat(absolute)

    async def listdir(self, path: Path) -> list[str]:
        absolute = self._policy.resolve_relative(path)
        return await self._storage.listdir(absolute)

    async def tree(self, path: Path, *, max_depth: int = 3, show_hidden: bool = False, exclude_patterns: Optional[List[str]] = None, file_types: Optional[List[str]] = None) -> list[str]:
        """Generate directory tree structure with filtering."""
        absolute = self._policy.resolve_relative(path)
        lines: list[str] = []
        exclude_patterns = exclude_patterns or []
        
        # Pre-compile exclude patterns for better performance
        compiled_exclude_patterns = [self._compile_pattern(pattern) for pattern in exclude_patterns]
        
        # Convert file_types to set for O(1) lookup
        file_types_set = set(file_types) if file_types else None

        async def _walk(current: Path, prefix: str, depth: int) -> None:
            if depth >= max_depth:
                return
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for i, entry in enumerate(entries):
                name = entry.name
                if not show_hidden and name.startswith('.'):
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
        return lines

    async def collect_all_files(self, path: Path, *, max_files: int = 100) -> list[Path]:
        """Recursively collect all files under the given path."""
        all_files = []
        
        async def _collect_files(current_path: Path):
            if len(all_files) >= max_files:
                return
            try:
                items = await self._storage.listdir(current_path)
                for item in items:
                    if len(all_files) >= max_files:
                        return
                    item_path = current_path / item
                    absolute_path = self._policy.resolve_relative(item_path)
                    if absolute_path.is_file():
                        all_files.append(item_path)
                    elif absolute_path.is_dir():
                        await _collect_files(item_path)
            except Exception:
                pass
        
        await _collect_files(path)
        return all_files

    async def search(self, path: Path, *, query: str, by: str = "name", file_types: Optional[List[str]] = None, case_sensitive: bool = False, max_results: int = 100) -> list[SearchResult]:
        """Search for files by name or content with performance optimizations."""
        absolute = self._policy.resolve_relative(path)
        results: list[SearchResult] = []
        
        # Performance optimizations
        file_types_set = set(file_types) if file_types else None
        search_query = query if case_sensitive else query.lower()
        max_matches_per_file = 50

        def _match_name(name: str) -> bool:
            """Check if filename matches query."""
            if not case_sensitive:
                return search_query in name.lower()
            return query in name

        async def _search_file(file_path: Path) -> Optional[SearchResult]:
            """Search within a single file."""
            if by == "name":
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
                    
                    hay = line if case_sensitive else line.lower()
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
                    if len(results) >= max_results:
                        return
                    if file_types_set and entry.suffix.lower() not in file_types_set:
                        continue
                    res = await _search_file(entry)
                    if res:
                        results.append(res)
                        if len(results) >= max_results:
                            return
            except Exception:
                pass  # Skip directories we can't access

        if absolute.is_file():
            single = await _search_file(absolute)
            return [single] if single else []

        await _walk(absolute)
        return results


