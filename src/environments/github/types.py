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


# Request/Result types for service layer

class CreateRepositoryRequest(BaseModel):
    """Request for creating a repository."""
    name: str = Field(..., description="Repository name")
    description: Optional[str] = Field(None, description="Repository description")
    private: bool = Field(False, description="Whether repository is private")
    auto_init: bool = Field(False, description="Whether to initialize with README")


class CreateRepositoryResult(BaseModel):
    """Result of creating a repository."""
    repository: Optional[GitHubRepository] = None
    success: bool
    message: str


class ForkRepositoryRequest(BaseModel):
    """Request for forking a repository."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")


class ForkRepositoryResult(BaseModel):
    """Result of forking a repository."""
    repository: Optional[GitHubRepository] = None
    success: bool
    message: str


class DeleteRepositoryRequest(BaseModel):
    """Request for deleting a repository."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")


class DeleteRepositoryResult(BaseModel):
    """Result of deleting a repository."""
    success: bool
    message: str


class GetRepositoryRequest(BaseModel):
    """Request for getting repository information."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")


class GetRepositoryResult(BaseModel):
    """Result of getting repository information."""
    repository: Optional[GitHubRepository] = None
    success: bool
    message: str


class CloneRepositoryRequest(BaseModel):
    """Request for cloning a repository."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    local_path: str = Field(..., description="Local path to clone to")
    branch: Optional[str] = Field(None, description="Branch to clone")


class CloneRepositoryResult(BaseModel):
    """Result of cloning a repository."""
    local_path: str
    success: bool
    message: str


class InitRepositoryRequest(BaseModel):
    """Request for initializing a repository."""
    local_path: str = Field(..., description="Local path to initialize")
    remote_url: Optional[str] = Field(None, description="Remote repository URL")


class InitRepositoryResult(BaseModel):
    """Result of initializing a repository."""
    local_path: str
    success: bool
    message: str


class GitCommitRequest(BaseModel):
    """Request for committing changes."""
    local_path: str = Field(..., description="Local repository path")
    message: str = Field(..., description="Commit message")
    files: Optional[List[str]] = Field(None, description="Specific files to commit")


class GitCommitResult(BaseModel):
    """Result of committing changes."""
    local_path: str
    commit_hash: Optional[str] = None
    success: bool
    message: str


class GitPushRequest(BaseModel):
    """Request for pushing changes."""
    local_path: str = Field(..., description="Local repository path")
    remote: str = Field("origin", description="Remote name")
    branch: str = Field("main", description="Branch to push")


class GitPushResult(BaseModel):
    """Result of pushing changes."""
    local_path: str
    success: bool
    message: str


class GitPullRequest(BaseModel):
    """Request for pulling changes."""
    local_path: str = Field(..., description="Local repository path")
    remote: str = Field("origin", description="Remote name")
    branch: str = Field("main", description="Branch to pull")


class GitPullResult(BaseModel):
    """Result of pulling changes."""
    local_path: str
    success: bool
    message: str


class GitFetchRequest(BaseModel):
    """Request for fetching changes."""
    local_path: str = Field(..., description="Local repository path")
    remote: str = Field("origin", description="Remote name")


class GitFetchResult(BaseModel):
    """Result of fetching changes."""
    local_path: str
    success: bool
    message: str


class GitCreateBranchRequest(BaseModel):
    """Request for creating a branch."""
    local_path: str = Field(..., description="Local repository path")
    branch_name: str = Field(..., description="New branch name")
    from_branch: Optional[str] = Field(None, description="Branch to create from")


class GitCreateBranchResult(BaseModel):
    """Result of creating a branch."""
    local_path: str
    branch_name: str
    success: bool
    message: str


class GitCheckoutBranchRequest(BaseModel):
    """Request for checking out a branch."""
    local_path: str = Field(..., description="Local repository path")
    branch_name: str = Field(..., description="Branch to checkout")


class GitCheckoutBranchResult(BaseModel):
    """Result of checking out a branch."""
    local_path: str
    branch_name: str
    success: bool
    message: str


class GitListBranchesRequest(BaseModel):
    """Request for listing branches."""
    local_path: str = Field(..., description="Local repository path")


class GitListBranchesResult(BaseModel):
    """Result of listing branches."""
    local_path: str
    branches: List[str] = Field(default_factory=list)
    current_branch: Optional[str] = None
    success: bool
    message: str


class GitDeleteBranchRequest(BaseModel):
    """Request for deleting a branch."""
    local_path: str = Field(..., description="Local repository path")
    branch_name: str = Field(..., description="Branch to delete")
    force: bool = Field(False, description="Force delete")


class GitDeleteBranchResult(BaseModel):
    """Result of deleting a branch."""
    local_path: str
    branch_name: str
    success: bool
    message: str


class GitStatusRequest(BaseModel):
    """Request for getting git status."""
    local_path: str = Field(..., description="Local repository path")


class GitStatusResult(BaseModel):
    """Result of getting git status."""
    local_path: str
    status: Optional[GitStatus] = None
    success: bool
    message: str
