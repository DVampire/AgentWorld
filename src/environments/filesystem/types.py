from __future__ import annotations

from pathlib import Path
from typing import Literal, NamedTuple, Optional, Union, List, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field, validator


class FilePath(NamedTuple):
    """Pair of relative and absolute paths under the sandbox base directory."""

    relative: Path
    absolute: Path


class FileReadRequest(BaseModel):
    """Request for reading file content with optional filtering."""
    path: Path
    as_text: bool = True
    encoding: str = "utf-8"
    start_line: Optional[int] = Field(None, ge=1, description="Start line number (1-based)")
    end_line: Optional[int] = Field(None, ge=1, description="End line number (1-based)")
    max_bytes: Optional[int] = Field(5 * 1024 * 1024, ge=0, description="Maximum bytes to read")

    @validator('end_line')
    def validate_line_range(cls, v, values):
        """Validate that end_line is greater than start_line."""
        if v is not None and 'start_line' in values and values['start_line'] is not None:
            if v <= values['start_line']:
                raise ValueError('end_line must be greater than start_line')
        return v


class FileReadResult(BaseModel):
    """Result of file read operation with content and metadata."""
    path: Path
    source: Literal["cache", "disk", "remote"]
    content_bytes: Optional[bytes] = None
    content_text: Optional[str] = None
    total_lines: Optional[int] = Field(None, ge=0)
    preview: Optional[str] = None
    read_time: Optional[datetime] = None
    file_size: Optional[int] = Field(None, ge=0)


class SearchMatch(BaseModel):
    """A single search match within a file."""
    line: int = Field(ge=1, description="Line number (1-based)")
    text: str
    column: Optional[int] = Field(None, ge=0, description="Column position of match")
    context_before: Optional[str] = None
    context_after: Optional[str] = None


class SearchResult(BaseModel):
    """Search result containing file path and matches."""
    path: Path
    matches: List[SearchMatch]
    total_matches: Optional[int] = Field(None, ge=0)
    search_time: Optional[datetime] = None


class FileStats(BaseModel):
    """File system statistics."""
    size: int = Field(ge=0)
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    accessed: Optional[datetime] = None
    permissions: Optional[str] = None
    is_directory: bool = False
    is_file: bool = False
    is_symlink: bool = False


class DirectoryInfo(BaseModel):
    """Directory information with contents summary."""
    path: Path
    total_files: int = Field(ge=0)
    total_directories: int = Field(ge=0)
    total_size: int = Field(ge=0)
    file_types: Dict[str, int] = Field(default_factory=dict)
    last_modified: Optional[datetime] = None


class CacheStats(BaseModel):
    """Cache performance statistics."""
    entries: int = Field(ge=0)
    total_bytes: int = Field(ge=0)
    max_entries: int = Field(ge=0)
    max_bytes: int = Field(ge=0)
    hits: int = Field(ge=0)
    misses: int = Field(ge=0)
    hit_rate: float = Field(ge=0.0, le=100.0)
    ttl_seconds: int = Field(ge=0)


