from __future__ import annotations

import json
import csv
import io
from typing import Optional, Protocol, Union, List, Dict, Any

from src.filesystem.types import FileReadRequest, FileReadResult


class ContentHandler(Protocol):
    """Protocol for content handlers that process different file types."""
    extensions: set[str]

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult: ...

    async def encode(self, text: Union[str, bytes], *, mode: str, encoding: str) -> bytes: ...


class TextHandler:
    """Handler for plain text files."""
    extensions: set[str] = {".txt", ".md", ".py", ".log", ".cfg", ".ini", ".conf", ".yml", ".yaml", ".xml", ".html", ".css", ".js", ".ts", ".sh", ".bat", ".ps1"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """Decode text content with line range support."""
        text = data.decode(request.encoding, errors="replace")
        total_lines: Optional[int] = None
        content_text = text
        
        if request.start_line is not None or request.end_line is not None:
            lines = text.splitlines()
            total_lines = len(lines)
            start = (request.start_line - 1) if request.start_line else 0
            end = request.end_line if request.end_line else total_lines
            if start < 0 or end > total_lines or start >= end:
                # Out-of-range returns empty slice
                content_text = ""
            else:
                content_text = "\n".join(lines[start:end])
        
        # Generate preview
        preview = None
        if content_text:
            plines = content_text.splitlines()[:3]
            preview = "\n".join(plines)
        
        return FileReadResult(
            path=request.path,
            source="disk",
            content_bytes=None,
            content_text=content_text,
            total_lines=total_lines,
            preview=preview,
        )

    async def encode(self, text: Union[str, bytes], *, mode: str, encoding: str) -> bytes:
        """Encode text content."""
        if isinstance(text, bytes):
            return text
        return text.encode(encoding)


class JsonHandler(TextHandler):
    """Handler for JSON files with structure-aware preview."""
    extensions: set[str] = {".json", ".jsonl"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """Decode JSON content with structure preview."""
        # Keep text for replace/search, but prepare preview from JSON structure
        base = await super().decode(data, request)
        try:
            text = data.decode(request.encoding, errors="replace")
            if request.path.suffix == ".jsonl":
                # Handle JSONL (JSON Lines) format
                lines = text.strip().split('\n')
                if lines and lines[0]:
                    first_obj = json.loads(lines[0])
                    base.preview = f"JSONL: {type(first_obj).__name__} with {len(lines)} lines"
            else:
                # Handle regular JSON
                obj = json.loads(text)
                if isinstance(obj, dict):
                    keys = list(obj.keys())[:5]
                    base.preview = f"JSON Object with keys: {', '.join(keys)}"
                elif isinstance(obj, list):
                    base.preview = f"JSON Array with {len(obj)} items"
                else:
                    base.preview = f"JSON {type(obj).__name__}"
        except Exception:
            # ignore JSON errors, fall back to text preview
            pass
        return base


class CsvHandler(TextHandler):
    """Handler for CSV files with structure-aware preview."""
    extensions: set[str] = {".csv", ".tsv"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """Decode CSV content with structure preview."""
        base = await super().decode(data, request)
        try:
            text = data.decode(request.encoding, errors="replace")
            delimiter = ',' if request.path.suffix == '.csv' else '\t'
            
            # Read first few lines to understand structure
            lines = text.splitlines()[:3]
            if lines:
                reader = csv.reader(lines, delimiter=delimiter)
                rows = list(reader)
                if rows:
                    headers = rows[0] if len(rows) > 0 else []
                    row_count = len(text.splitlines())
                    base.preview = f"CSV: {len(headers)} columns, {row_count} rows. Headers: {', '.join(headers[:3])}"
        except Exception:
            # ignore CSV errors, fall back to text preview
            pass
        return base


class BinaryHandler:
    """Handler for binary files."""
    extensions: set[str] = {".bin", ".dat", ".exe", ".dll", ".so", ".dylib", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".ico", ".svg", ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """Decode binary content with hex preview."""
        # For binary, do not attempt to decode text
        preview = f"Binary file ({len(data)} bytes): {data[:32].hex()}"
        return FileReadResult(
            path=request.path,
            source="disk",
            content_bytes=data,
            content_text=None,
            total_lines=None,
            preview=preview,
        )

    async def encode(self, text: Union[str, bytes], *, mode: str, encoding: str) -> bytes:
        """Encode binary content."""
        if isinstance(text, bytes):
            return text
        return text.encode(encoding)


class MarkdownHandler(TextHandler):
    """Handler for Markdown files with enhanced preview."""
    extensions: set[str] = {".md", ".markdown", ".mdown", ".mkdn", ".mkd", ".mdwn", ".mdtxt", ".mdtext"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """Decode Markdown content with structure preview."""
        base = await super().decode(data, request)
        try:
            text = data.decode(request.encoding, errors="replace")
            lines = text.splitlines()
            
            # Extract headers for preview
            headers = []
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if line.startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    title = line.lstrip('#').strip()
                    headers.append(f"{'  ' * (level-1)}- {title}")
                    if len(headers) >= 3:
                        break
            
            if headers:
                base.preview = f"Markdown with headers:\n" + "\n".join(headers)
        except Exception:
            pass
        return base


class PythonHandler(TextHandler):
    """Handler for Python files with code structure preview."""
    extensions: set[str] = {".py", ".pyi", ".pyc", ".pyo"}

    async def decode(self, data: bytes, request: FileReadRequest) -> FileReadResult:
        """Decode Python content with structure preview."""
        base = await super().decode(data, request)
        try:
            text = data.decode(request.encoding, errors="replace")
            lines = text.splitlines()
            
            # Extract class and function definitions
            definitions = []
            for i, line in enumerate(lines[:20]):  # Check first 20 lines
                line = line.strip()
                if line.startswith('class ') and ':' in line:
                    class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                    definitions.append(f"class {class_name}")
                elif line.startswith('def ') and ':' in line:
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    definitions.append(f"def {func_name}")
                elif line.startswith('async def ') and ':' in line:
                    func_name = line.split('async def ')[1].split('(')[0].strip()
                    definitions.append(f"async def {func_name}")
                
                if len(definitions) >= 5:
                    break
            
            if definitions:
                base.preview = f"Python code with:\n" + "\n".join(definitions[:5])
        except Exception:
            pass
        return base


class HandlerRegistry:
    """Registry for content handlers with priority-based selection."""
    
    def __init__(self) -> None:
        self._handlers: List[ContentHandler] = []
        self._extension_map: Dict[str, ContentHandler] = {}

    def register(self, handler: ContentHandler) -> None:
        """Register a content handler."""
        self._handlers.append(handler)
        # Update extension mapping
        for ext in handler.extensions:
            self._extension_map[ext.lower()] = handler

    def find_for_extension(self, suffix: str) -> Optional[ContentHandler]:
        """Find handler for file extension with O(1) lookup."""
        return self._extension_map.get(suffix.lower())

    def get_all_handlers(self) -> List[ContentHandler]:
        """Get all registered handlers."""
        return self._handlers.copy()

    def get_supported_extensions(self) -> set[str]:
        """Get all supported file extensions."""
        return set(self._extension_map.keys())


