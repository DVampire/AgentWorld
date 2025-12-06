"""GitHub data types."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


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


# Request types for service layer

class CreateRepositoryRequest(BaseModel):
    """Request for creating a repository."""
    name: str = Field(..., description="Repository name")
    description: Optional[str] = Field(None, description="Repository description")
    private: bool = Field(False, description="Whether repository is private")
    auto_init: bool = Field(False, description="Whether to initialize with README")


class ForkRepositoryRequest(BaseModel):
    """Request for forking a repository."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")


class DeleteRepositoryRequest(BaseModel):
    """Request for deleting a repository."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")


class GetRepositoryRequest(BaseModel):
    """Request for getting repository information."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")


class CloneRepositoryRequest(BaseModel):
    """Request for cloning a repository."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    local_path: str = Field(..., description="Local path to clone to")
    branch: Optional[str] = Field(None, description="Branch to clone")


class InitRepositoryRequest(BaseModel):
    """Request for initializing a repository."""
    local_path: str = Field(..., description="Local path to initialize")
    remote_url: Optional[str] = Field(None, description="Remote repository URL")


class GitCommitRequest(BaseModel):
    """Request for committing changes."""
    local_path: str = Field(..., description="Local repository path")
    message: str = Field(..., description="Commit message")
    files: Optional[List[str]] = Field(None, description="Specific files to commit")


class GitPushRequest(BaseModel):
    """Request for pushing changes."""
    local_path: str = Field(..., description="Local repository path")
    remote: str = Field("origin", description="Remote name")
    branch: str = Field("main", description="Branch to push")


class GitPullRequest(BaseModel):
    """Request for pulling changes."""
    local_path: str = Field(..., description="Local repository path")
    remote: str = Field("origin", description="Remote name")
    branch: str = Field("main", description="Branch to pull")


class GitFetchRequest(BaseModel):
    """Request for fetching changes."""
    local_path: str = Field(..., description="Local repository path")
    remote: str = Field("origin", description="Remote name")


class GitCreateBranchRequest(BaseModel):
    """Request for creating a branch."""
    local_path: str = Field(..., description="Local repository path")
    branch_name: str = Field(..., description="New branch name")
    from_branch: Optional[str] = Field(None, description="Branch to create from")


class GitCheckoutBranchRequest(BaseModel):
    """Request for checking out a branch."""
    local_path: str = Field(..., description="Local repository path")
    branch_name: str = Field(..., description="Branch to checkout")


class GitListBranchesRequest(BaseModel):
    """Request for listing branches."""
    local_path: str = Field(..., description="Local repository path")


class GitDeleteBranchRequest(BaseModel):
    """Request for deleting a branch."""
    local_path: str = Field(..., description="Local repository path")
    branch_name: str = Field(..., description="Branch to delete")
    force: bool = Field(False, description="Force delete")


class GitStatusRequest(BaseModel):
    """Request for getting git status."""
    local_path: str = Field(..., description="Local repository path")
