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


# Request/Result types for service layer

class FileWriteRequest(BaseModel):
    """Request for writing file content."""
    path: Path
    content: str
    mode: str = Field("w", description="Write mode: 'w' for overwrite, 'a' for append")
    encoding: str = Field("utf-8", description="Text encoding")


class FileWriteResult(BaseModel):
    """Result of file write operation."""
    path: Path
    bytes_written: int = Field(ge=0)
    success: bool
    message: str


class FileReplaceRequest(BaseModel):
    """Request for replacing text in a file."""
    path: Path
    old_string: str
    new_string: str
    start_line: Optional[int] = Field(None, ge=1, description="Start line number (1-based)")
    end_line: Optional[int] = Field(None, ge=1, description="End line number (1-based)")
    encoding: str = Field("utf-8", description="Text encoding")


class FileReplaceResult(BaseModel):
    """Result of file replace operation."""
    path: Path
    replacements_made: int = Field(ge=0)
    success: bool
    message: str


class FileDeleteRequest(BaseModel):
    """Request for deleting a file."""
    path: Path


class FileDeleteResult(BaseModel):
    """Result of file delete operation."""
    path: Path
    success: bool
    message: str


class FileCopyRequest(BaseModel):
    """Request for copying a file."""
    src_path: Path
    dst_path: Path
    overwrite: bool = Field(False, description="Whether to overwrite existing file")


class FileCopyResult(BaseModel):
    """Result of file copy operation."""
    src_path: Path
    dst_path: Path
    bytes_copied: int = Field(ge=0)
    success: bool
    message: str


class FileMoveRequest(BaseModel):
    """Request for moving a file."""
    src_path: Path
    dst_path: Path
    overwrite: bool = Field(False, description="Whether to overwrite existing file")


class FileMoveResult(BaseModel):
    """Result of file move operation."""
    src_path: Path
    dst_path: Path
    success: bool
    message: str


class DirectoryCreateRequest(BaseModel):
    """Request for creating a directory."""
    path: Path
    parents: bool = Field(True, description="Whether to create parent directories")


class DirectoryCreateResult(BaseModel):
    """Result of directory create operation."""
    path: Path
    success: bool
    message: str


class DirectoryDeleteRequest(BaseModel):
    """Request for deleting a directory."""
    path: Path
    recursive: bool = Field(False, description="Whether to delete recursively")


class DirectoryDeleteResult(BaseModel):
    """Result of directory delete operation."""
    path: Path
    success: bool
    message: str


class FileListRequest(BaseModel):
    """Request for listing directory contents."""
    path: Path
    show_hidden: bool = Field(False, description="Whether to show hidden files")
    file_types: Optional[List[str]] = Field(None, description="Filter by file extensions")


class FileListResult(BaseModel):
    """Result of file list operation."""
    path: Path
    files: List[str] = Field(default_factory=list)
    directories: List[str] = Field(default_factory=list)
    total_files: int = Field(ge=0)
    total_directories: int = Field(ge=0)


class FileTreeRequest(BaseModel):
    """Request for generating directory tree."""
    path: Path
    max_depth: int = Field(3, ge=1, le=10, description="Maximum tree depth")
    show_hidden: bool = Field(False, description="Whether to show hidden files")
    exclude_patterns: Optional[List[str]] = Field(None, description="Patterns to exclude")
    file_types: Optional[List[str]] = Field(None, description="Filter by file extensions")


class FileTreeResult(BaseModel):
    """Result of file tree operation."""
    path: Path
    tree_lines: List[str] = Field(default_factory=list)
    total_files: int = Field(ge=0)
    total_directories: int = Field(ge=0)


class FileSearchRequest(BaseModel):
    """Request for searching files."""
    path: Path
    query: str
    by: str = Field("name", description="Search by 'name' or 'content'")
    file_types: Optional[List[str]] = Field(None, description="Filter by file extensions")
    case_sensitive: bool = Field(False, description="Whether search is case sensitive")
    max_results: int = Field(100, ge=1, le=1000, description="Maximum number of results")


class FileSearchResult(BaseModel):
    """Result of file search operation."""
    query: str
    search_by: str
    results: List[SearchResult] = Field(default_factory=list)
    total_found: int = Field(ge=0)
    search_time: Optional[datetime] = None


class FileStatRequest(BaseModel):
    """Request for getting file statistics."""
    path: Path


class FileStatResult(BaseModel):
    """Result of file stat operation."""
    path: Path
    stats: Optional[FileStats] = None
    exists: bool
    success: bool
    message: str


