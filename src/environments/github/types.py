"""Type definitions for GitHub service."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from pathlib import Path

from pydantic import BaseModel, Field, validator


class GitHubUser(BaseModel):
    """GitHub user information."""
    id: int
    login: str
    name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    html_url: Optional[str] = None
    type: Literal["User", "Organization"] = "User"
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    following: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class GitHubOrganization(BaseModel):
    """GitHub organization information."""
    id: int
    login: str
    name: Optional[str] = None
    description: Optional[str] = None
    avatar_url: Optional[str] = None
    html_url: Optional[str] = None
    type: Literal["Organization"] = "Organization"
    public_repos: Optional[int] = None
    followers: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class GitHubRepository(BaseModel):
    """GitHub repository information."""
    id: int
    name: str
    full_name: str
    description: Optional[str] = None
    private: bool = False
    html_url: Optional[str] = None
    clone_url: Optional[str] = None
    ssh_url: Optional[str] = None
    default_branch: str = "main"
    language: Optional[str] = None
    languages: Optional[Dict[str, int]] = None
    stargazers_count: int = 0
    watchers_count: int = 0
    forks_count: int = 0
    open_issues_count: int = 0
    size: int = 0
    owner: Union[GitHubUser, GitHubOrganization]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    pushed_at: Optional[datetime] = None


class GitHubBranch(BaseModel):
    """GitHub branch information."""
    name: str
    commit: Optional[Dict[str, Any]] = None
    protected: bool = False
    protection_url: Optional[str] = None


class GitHubCommit(BaseModel):
    """GitHub commit information."""
    sha: str
    message: str
    author: Optional[GitHubUser] = None
    committer: Optional[GitHubUser] = None
    html_url: Optional[str] = None
    stats: Optional[Dict[str, int]] = None
    files: Optional[List[Dict[str, Any]]] = None
    parents: Optional[List[Dict[str, str]]] = None
    date: Optional[datetime] = None


class GitHubIssue(BaseModel):
    """GitHub issue information."""
    id: int
    number: int
    title: str
    body: Optional[str] = None
    state: Literal["open", "closed"] = "open"
    labels: List[str] = Field(default_factory=list)
    assignees: List[GitHubUser] = Field(default_factory=list)
    user: Optional[GitHubUser] = None
    html_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None


class GitHubPullRequest(BaseModel):
    """GitHub pull request information."""
    id: int
    number: int
    title: str
    body: Optional[str] = None
    state: Literal["open", "closed", "merged"] = "open"
    draft: bool = False
    head: Optional[Dict[str, Any]] = None
    base: Optional[Dict[str, Any]] = None
    user: Optional[GitHubUser] = None
    assignees: List[GitHubUser] = Field(default_factory=list)
    labels: List[str] = Field(default_factory=list)
    html_url: Optional[str] = None
    diff_url: Optional[str] = None
    patch_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None


class GitHubFile(BaseModel):
    """GitHub file content."""
    name: str
    path: str
    sha: str
    size: int
    url: Optional[str] = None
    html_url: Optional[str] = None
    download_url: Optional[str] = None
    type: Literal["file"] = "file"
    content: Optional[str] = None
    encoding: Optional[str] = None


class GitHubDirectory(BaseModel):
    """GitHub directory content."""
    name: str
    path: str
    sha: str
    size: int
    url: Optional[str] = None
    html_url: Optional[str] = None
    type: Literal["dir"] = "dir"


class GitHubContent(BaseModel):
    """GitHub content (file or directory)."""
    name: str
    path: str
    sha: str
    size: int
    url: Optional[str] = None
    html_url: Optional[str] = None
    download_url: Optional[str] = None
    type: Literal["file", "dir", "symlink", "submodule"]
    content: Optional[str] = None
    encoding: Optional[str] = None


class GitHubSearchResult(BaseModel):
    """GitHub search result."""
    total_count: int
    incomplete_results: bool
    items: List[Union[GitHubRepository, GitHubIssue, GitHubPullRequest, GitHubUser]]


class GitHubRequest(BaseModel):
    """GitHub API request parameters."""
    endpoint: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    params: Optional[Dict[str, Any]] = None
    data: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: Optional[int] = 30


class GitHubResponse(BaseModel):
    """GitHub API response."""
    status_code: int
    headers: Dict[str, str]
    data: Optional[Any] = None
    error: Optional[str] = None
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None


class GitHubAuth(BaseModel):
    """GitHub authentication configuration."""
    token: str
    username: Optional[str] = None
    base_url: str = "https://api.github.com"
    timeout: int = 30
    
    @validator('token')
    def validate_token(cls, v):
        """Validate that token is not empty."""
        if not v or not v.strip():
            raise ValueError('GitHub token cannot be empty')
        return v.strip()
    
    @validator('base_url')
    def validate_base_url(cls, v):
        """Validate base URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Base URL must start with http:// or https://')
        return v.rstrip('/')


class GitHubConfig(BaseModel):
    """GitHub service configuration."""
    auth: GitHubAuth
    default_per_page: int = Field(default=30, ge=1, le=100)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0.0)
    cache_ttl: int = Field(default=300, ge=0)  # 5 minutes
    enable_cache: bool = True
    enable_rate_limit: bool = True
