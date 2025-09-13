"""GitHub data types."""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime


@dataclass
class GitHubUser:
    """GitHub user information."""
    login: str
    id: int
    name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None
    html_url: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class GitHubRepository:
    """GitHub repository information."""
    full_name: str
    name: str
    owner: str
    description: Optional[str] = None
    private: bool = False
    html_url: Optional[str] = None
    clone_url: Optional[str] = None
    ssh_url: Optional[str] = None
    language: Optional[str] = None
    stargazers_count: int = 0
    forks_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class GitHubBranch:
    """GitHub branch information."""
    name: str
    sha: str
    protected: bool = False
    commit_url: Optional[str] = None


@dataclass
class GitStatus:
    """Git repository status."""
    is_dirty: bool
    untracked_files: List[str]
    modified_files: List[str]
    staged_files: List[str]
    current_branch: str
    branches: List[str]
