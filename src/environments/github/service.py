"""GitHub service implementation using PyGithub + GitPython."""
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

from github import Github, GithubException
from git import Repo, InvalidGitRepositoryError, GitCommandError

from src.environments.github.types import (
    GitHubRepository,
    GitHubUser, 
    GitHubBranch,
    GitStatus,
    CreateRepositoryRequest, 
    CreateRepositoryResult,
    ForkRepositoryRequest, 
    ForkRepositoryResult,
    DeleteRepositoryRequest, 
    DeleteRepositoryResult,
    GetRepositoryRequest, 
    GetRepositoryResult,
    CloneRepositoryRequest,
    CloneRepositoryResult,
    InitRepositoryRequest, 
    InitRepositoryResult,
    GitCommitRequest, 
    GitCommitResult,
    GitPushRequest,
    GitPushResult,
    GitPullRequest, 
    GitPullResult,
    GitFetchRequest,
    GitFetchResult,
    GitCreateBranchRequest, 
    GitCreateBranchResult,
    GitCheckoutBranchRequest,
    GitCheckoutBranchResult,
    GitListBranchesRequest,
    GitListBranchesResult,
    GitDeleteBranchRequest,
    GitDeleteBranchResult,
    GitStatusRequest,
    GitStatusResult
)
from src.environments.github.exceptions import (
    GitHubError, 
    AuthenticationError, 
    NotFoundError, 
    GitError, 
    RepositoryError
)


class GitHubService:
    """GitHub service using PyGithub + GitPython."""

    def __init__(self, token: str, username: Optional[str] = None):
        """Initialize GitHub service.
        
        Args:
            token: GitHub Personal Access Token
            username: GitHub username (optional)
        """
        self.token = token
        self.username = username
        self._github: Optional[Github] = None
        self._authenticated_user: Optional[GitHubUser] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def initialize(self) -> None:
        """Initialize the GitHub service."""
        try:
            self._github = Github(self.token)
            github_user = self._github.get_user()
            
            self._authenticated_user = GitHubUser(
                login=github_user.login,
                id=github_user.id,
                name=github_user.name,
                email=github_user.email,
                avatar_url=github_user.avatar_url,
                html_url=github_user.html_url,
                created_at=github_user.created_at
            )
            
            if self.username is None:
                self.username = self._authenticated_user.login
                
        except GithubException as e:
            if e.status == 401:
                raise AuthenticationError(f"Invalid GitHub token: {e}")
            raise GitHubError(f"Failed to initialize GitHub service: {e}")
        except Exception as e:
            raise GitHubError(f"Failed to initialize GitHub service: {e}")

    async def cleanup(self) -> None:
        """Cleanup the GitHub service."""
        self._github = None
        self._authenticated_user = None

    @property
    def github(self) -> Github:
        """Get GitHub instance."""
        if self._github is None:
            raise RuntimeError("GitHub service not initialized")
        return self._github

    @property
    def authenticated_user(self) -> GitHubUser:
        """Get authenticated user."""
        if self._authenticated_user is None:
            raise RuntimeError("GitHub service not initialized")
        return self._authenticated_user

    # --------------- Repository Operations ---------------

    async def create_repository(self, request: CreateRepositoryRequest) -> CreateRepositoryResult:
        """Create a new GitHub repository."""
        try:
            # Get the PyGithub user object for creating repositories
            github_user = await asyncio.to_thread(self.github.get_user)
            # Use asyncio.to_thread to run the synchronous GitHub API call in a thread pool
            repo = await asyncio.to_thread(
                github_user.create_repo,
                name=request.name,
                description=request.description,
                private=request.private,
                auto_init=request.auto_init
            )
            
            repository = GitHubRepository(
                full_name=repo.full_name,
                name=repo.name,
                owner=repo.owner.login,
                description=repo.description,
                private=repo.private,
                html_url=repo.html_url,
                clone_url=repo.clone_url,
                ssh_url=repo.ssh_url,
                language=repo.language,
                stargazers_count=repo.stargazers_count,
                forks_count=repo.forks_count,
                created_at=repo.created_at,
                updated_at=repo.updated_at
            )
            
            return CreateRepositoryResult(
                repository=repository,
                success=True,
                message=f"Successfully created repository {request.name}"
            )
            
        except GithubException as e:
            error_msg = f"Failed to create repository '{request.name}': {e}"
            if e.status == 401:
                error_msg = "Invalid GitHub token or insufficient permissions"
            elif e.status == 403:
                error_msg = f"Permission denied: Cannot create repository '{request.name}'"
            elif e.status == 422:
                error_msg = f"Repository '{request.name}' already exists or invalid name"
            elif e.status == 404:
                error_msg = "User not found or repository creation failed"
            
            return CreateRepositoryResult(
                repository=None,
                success=False,
                message=error_msg
            )
        except Exception as e:
            return CreateRepositoryResult(
                repository=None,
                success=False,
                message=f"Failed to create repository '{request.name}': {e}"
            )

    async def fork_repository(self, request: ForkRepositoryRequest) -> ForkRepositoryResult:
        """Fork a repository to your account."""
        try:
            # Get the original repository
            original_repo = await asyncio.to_thread(self.github.get_repo, f"{request.owner}/{request.repo}")
            
            # Fork the repository
            github_user = await asyncio.to_thread(self.github.get_user)
            forked_repo = await asyncio.to_thread(github_user.create_fork, original_repo)
            
            repository = GitHubRepository(
                full_name=forked_repo.full_name,
                name=forked_repo.name,
                owner=forked_repo.owner.login,
                description=forked_repo.description,
                private=forked_repo.private,
                html_url=forked_repo.html_url,
                clone_url=forked_repo.clone_url,
                ssh_url=forked_repo.ssh_url,
                language=forked_repo.language,
                stargazers_count=forked_repo.stargazers_count,
                forks_count=forked_repo.forks_count,
                created_at=forked_repo.created_at,
                updated_at=forked_repo.updated_at
            )
            
            return ForkRepositoryResult(
                repository=repository,
                success=True,
                message=f"Successfully forked repository {request.owner}/{request.repo}"
            )
            
        except GithubException as e:
            error_msg = f"Failed to fork repository '{request.owner}/{request.repo}': {e}"
            if e.status == 422:
                error_msg = f"Repository '{request.owner}/{request.repo}' cannot be forked (may already be forked or private)"
            elif e.status == 404:
                error_msg = f"Repository '{request.owner}/{request.repo}' not found"
            
            return ForkRepositoryResult(
                repository=None,
                success=False,
                message=error_msg
            )
        except Exception as e:
            return ForkRepositoryResult(
                repository=None,
                success=False,
                message=f"Failed to fork repository '{request.owner}/{request.repo}': {e}"
            )

    async def delete_repository(self, request: DeleteRepositoryRequest) -> DeleteRepositoryResult:
        """Delete a repository."""
        try:
            repository = await asyncio.to_thread(self.github.get_repo, f"{request.owner}/{request.repo}")
            await asyncio.to_thread(repository.delete)
            
            return DeleteRepositoryResult(
                success=True,
                message=f"Successfully deleted repository {request.owner}/{request.repo}"
            )
            
        except GithubException as e:
            error_msg = f"Failed to delete repository '{request.owner}/{request.repo}': {e}"
            if e.status == 404:
                error_msg = f"Repository '{request.owner}/{request.repo}' not found"
            elif e.status == 403:
                error_msg = f"Permission denied: Cannot delete repository '{request.owner}/{request.repo}'"
            
            return DeleteRepositoryResult(
                success=False,
                message=error_msg
            )
        except Exception as e:
            return DeleteRepositoryResult(
                success=False,
                message=f"Failed to delete repository '{request.owner}/{request.repo}': {e}"
            )

    async def get_repository(self, request: GetRepositoryRequest) -> GetRepositoryResult:
        """Get repository information."""
        try:
            repository = await asyncio.to_thread(self.github.get_repo, f"{request.owner}/{request.repo}")
            
            repo_info = GitHubRepository(
                full_name=repository.full_name,
                name=repository.name,
                owner=repository.owner.login,
                description=repository.description,
                private=repository.private,
                html_url=repository.html_url,
                clone_url=repository.clone_url,
                ssh_url=repository.ssh_url,
                language=repository.language,
                stargazers_count=repository.stargazers_count,
                forks_count=repository.forks_count,
                created_at=repository.created_at,
                updated_at=repository.updated_at
            )
            
            return GetRepositoryResult(
                repository=repo_info,
                success=True,
                message=f"Successfully retrieved repository {request.owner}/{request.repo}"
            )
            
        except GithubException as e:
            error_msg = f"Failed to get repository '{request.owner}/{request.repo}': {e}"
            if e.status == 404:
                error_msg = f"Repository '{request.owner}/{request.repo}' not found"
            
            return GetRepositoryResult(
                repository=None,
                success=False,
                message=error_msg
            )
        except Exception as e:
            return GetRepositoryResult(
                repository=None,
                success=False,
                message=f"Failed to get repository '{request.owner}/{request.repo}': {e}"
            )

    async def clone_repository(self, request: CloneRepositoryRequest) -> CloneRepositoryResult:
        """Clone a repository to local directory."""
        try:
            repo_url = f"https://github.com/{request.owner}/{request.repo}.git"
            local_path = Path(request.local_path)
            
            if local_path.exists():
                return CloneRepositoryResult(
                    local_path=request.local_path,
                    success=False,
                    message=f"Directory '{request.local_path}' already exists"
                )
            
            # Clone repository
            if request.branch:
                Repo.clone_from(repo_url, local_path, branch=request.branch)
                return CloneRepositoryResult(
                    local_path=request.local_path,
                    success=True,
                    message=f"Repository '{request.owner}/{request.repo}' cloned to '{request.local_path}' (branch: {request.branch})"
                )
            else:
                Repo.clone_from(repo_url, local_path)
                return CloneRepositoryResult(
                    local_path=request.local_path,
                    success=True,
                    message=f"Repository '{request.owner}/{request.repo}' cloned to '{request.local_path}'"
                )
                
        except GitCommandError as e:
            return CloneRepositoryResult(
                local_path=request.local_path,
                success=False,
                message=f"Failed to clone repository '{request.owner}/{request.repo}': {str(e)}"
            )
        except Exception as e:
            return CloneRepositoryResult(
                local_path=request.local_path,
                success=False,
                message=f"Failed to clone repository '{request.owner}/{request.repo}': {str(e)}"
            )

    async def init_repository(self, request: InitRepositoryRequest) -> InitRepositoryResult:
        """Initialize a local directory as Git repository."""
        try:
            local_path = Path(request.local_path)
            
            if not local_path.exists():
                return InitRepositoryResult(
                    local_path=request.local_path,
                    success=False,
                    message=f"Directory '{request.local_path}' does not exist"
                )
            
            if (local_path / '.git').exists():
                return InitRepositoryResult(
                    local_path=request.local_path,
                    success=False,
                    message=f"Directory '{request.local_path}' is already a Git repository"
                )
            
            # Initialize repository
            repo = Repo.init(local_path)
            
            message = f"Git repository initialized in '{request.local_path}'"
            
            if request.remote_url:
                try:
                    repo.create_remote('origin', request.remote_url)
                    message += f" with remote origin: {request.remote_url}"
                except Exception as e:
                    message += f" (failed to add remote: {str(e)})"
            
            return InitRepositoryResult(
                local_path=request.local_path,
                success=True,
                message=message
            )
            
        except Exception as e:
            return InitRepositoryResult(
                local_path=request.local_path,
                success=False,
                message=f"Failed to initialize repository in '{request.local_path}': {str(e)}"
            )

    # --------------- Git Operations ---------------

    async def git_commit(self, request: GitCommitRequest) -> GitCommitResult:
        """Commit changes to local repository."""
        try:
            repo = Repo(request.local_path)
            
            # Add files
            if request.files is None:  # add_all
                repo.git.add(A=True)
            elif request.files:
                for file in request.files:
                    repo.git.add(file)
            
            # Check if there are changes to commit
            if not repo.is_dirty() and not repo.untracked_files:
                return GitCommitResult(
                    local_path=request.local_path,
                    commit_hash=None,
                    success=True,
                    message="No changes to commit"
                )
            
            # Commit changes
            commit = repo.index.commit(request.message)
            return GitCommitResult(
                local_path=request.local_path,
                commit_hash=commit.hexsha,
                success=True,
                message=f"Commit created: {commit.hexsha[:8]} - {request.message}"
            )
            
        except InvalidGitRepositoryError:
            return GitCommitResult(
                local_path=request.local_path,
                commit_hash=None,
                success=False,
                message=f"'{request.local_path}' is not a valid Git repository"
            )
        except GitCommandError as e:
            return GitCommitResult(
                local_path=request.local_path,
                commit_hash=None,
                success=False,
                message=f"Failed to commit in '{request.local_path}': {str(e)}"
            )
        except Exception as e:
            return GitCommitResult(
                local_path=request.local_path,
                commit_hash=None,
                success=False,
                message=f"Failed to commit in '{request.local_path}': {str(e)}"
            )

    async def git_push(self, request: GitPushRequest) -> GitPushResult:
        """Push changes to remote repository."""
        try:
            repo = Repo(request.local_path)
            
            branch = request.branch
            if branch is None:
                branch = repo.active_branch.name
            
            # Push to remote
            origin = repo.remote(request.remote)
            origin.push(branch)
            
            return GitPushResult(
                local_path=request.local_path,
                success=True,
                message=f"Successfully pushed branch '{branch}' to remote '{request.remote}'"
            )
            
        except InvalidGitRepositoryError:
            return GitPushResult(
                local_path=request.local_path,
                success=False,
                message=f"'{request.local_path}' is not a valid Git repository"
            )
        except GitCommandError as e:
            return GitPushResult(
                local_path=request.local_path,
                success=False,
                message=f"Failed to push from '{request.local_path}': {str(e)}"
            )
        except Exception as e:
            return GitPushResult(
                local_path=request.local_path,
                success=False,
                message=f"Failed to push from '{request.local_path}': {str(e)}"
            )

    async def git_pull(self, request: GitPullRequest) -> GitPullResult:
        """Pull changes from remote repository.
        
        Args:
            local_path: Local repository path
            remote: Remote name (default: origin)
            branch: Branch name (optional, uses current branch if not specified)
            
        Returns:
            str: Pull result message
        """
        try:
            repo = Repo(request.local_path)
            
            branch = request.branch
            if branch is None:
                branch = repo.active_branch.name
            
            # Pull from remote
            origin = repo.remote(request.remote)
            origin.pull(branch)
            
            return GitPullResult(
                local_path=request.local_path,
                success=True,
                message=f"Successfully pulled branch '{branch}' from remote '{request.remote}'"
            )
            
        except InvalidGitRepositoryError:
            return GitPullResult(
                local_path=request.local_path,
                success=False,
                message=f"'{request.local_path}' is not a valid Git repository"
            )
        except GitCommandError as e:
            return GitPullResult(
                local_path=request.local_path,
                success=False,
                message=f"Failed to pull to '{request.local_path}': {str(e)}"
            )
        except Exception as e:
            return GitPullResult(
                local_path=request.local_path,
                success=False,
                message=f"Failed to pull to '{request.local_path}': {str(e)}"
            )

    async def git_fetch(self, request: GitFetchRequest) -> GitFetchResult:
        """Fetch changes from remote repository.
        
        Args:
            local_path: Local repository path
            remote: Remote name (default: origin)
            
        Returns:
            str: Fetch result message
        """
        try:
            repo = Repo(request.local_path)
            
            # Fetch from remote
            origin = repo.remote(request.remote)
            origin.fetch()
            
            return GitFetchResult(
                local_path=request.local_path,
                success=True,
                message=f"Successfully fetched from remote '{request.remote}'"
            )
            
        except InvalidGitRepositoryError:
            return GitFetchResult(
                local_path=request.local_path,
                success=False,
                message=f"'{request.local_path}' is not a valid Git repository"
            )
        except GitCommandError as e:
            return GitFetchResult(
                local_path=request.local_path,
                success=False,
                message=f"Failed to fetch to '{request.local_path}': {str(e)}"
            )
        except Exception as e:
            return GitFetchResult(
                local_path=request.local_path,
                success=False,
                message=f"Failed to fetch to '{request.local_path}': {str(e)}"
            )

    # --------------- Branch Operations ---------------

    async def git_create_branch(
        self,
        local_path: str,
        branch_name: str,
        checkout: bool = True
    ) -> str:
        """Create a new branch.
        
        Args:
            local_path: Local repository path
            branch_name: New branch name
            checkout: Whether to checkout the new branch
            
        Returns:
            str: Branch creation result message
        """
        try:
            repo = Repo(local_path)
            
            # Check if branch already exists
            if branch_name in [branch.name for branch in repo.branches]:
                return f"Branch '{branch_name}' already exists"
            
            # Create new branch
            new_branch = repo.create_head(branch_name)
            
            if checkout:
                new_branch.checkout()
                return f"Branch '{branch_name}' created and checked out"
            else:
                return f"Branch '{branch_name}' created"
                
        except InvalidGitRepositoryError:
            raise GitError(f"'{local_path}' is not a valid Git repository")
        except GitCommandError as e:
            raise GitError(f"Failed to create branch '{branch_name}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to create branch '{branch_name}': {str(e)}")

    async def git_checkout_branch(
        self,
        local_path: str,
        branch_name: str
    ) -> str:
        """Checkout an existing branch.
        
        Args:
            local_path: Local repository path
            branch_name: Branch name to checkout
            
        Returns:
            str: Checkout result message
        """
        try:
            repo = Repo(local_path)
            
            # Check if branch exists
            if branch_name not in [branch.name for branch in repo.branches]:
                return f"Branch '{branch_name}' does not exist"
            
            # Checkout branch
            repo.git.checkout(branch_name)
            
            return f"Checked out branch '{branch_name}'"
            
        except InvalidGitRepositoryError:
            raise GitError(f"'{local_path}' is not a valid Git repository")
        except GitCommandError as e:
            raise GitError(f"Failed to checkout branch '{branch_name}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to checkout branch '{branch_name}': {str(e)}")

    async def git_list_branches(self, local_path: str) -> str:
        """List all branches.
        
        Args:
            local_path: Local repository path
            
        Returns:
            str: List of branches
        """
        try:
            repo = Repo(local_path)
            
            branches = []
            current_branch = repo.active_branch.name
            
            for branch in repo.branches:
                status = " (current)" if branch.name == current_branch else ""
                branches.append(f"- {branch.name}{status}")
            
            if not branches:
                return "No branches found"
            
            return f"Branches:\n" + "\n".join(branches)
            
        except InvalidGitRepositoryError:
            raise GitError(f"'{local_path}' is not a valid Git repository")
        except Exception as e:
            raise GitError(f"Failed to list branches: {str(e)}")

    async def git_delete_branch(
        self,
        local_path: str,
        branch_name: str,
        force: bool = False
    ) -> str:
        """Delete a branch.
        
        Args:
            local_path: Local repository path
            branch_name: Branch name to delete
            force: Force delete even if not merged
            
        Returns:
            str: Delete result message
        """
        try:
            repo = Repo(local_path)
            
            # Check if branch exists
            if branch_name not in [branch.name for branch in repo.branches]:
                return f"Branch '{branch_name}' does not exist"
            
            # Check if trying to delete current branch
            if branch_name == repo.active_branch.name:
                return f"Cannot delete current branch '{branch_name}'. Switch to another branch first."
            
            # Delete branch
            if force:
                repo.git.branch('-D', branch_name)
                return f"Branch '{branch_name}' force deleted"
            else:
                repo.git.branch('-d', branch_name)
                return f"Branch '{branch_name}' deleted"
                
        except InvalidGitRepositoryError:
            raise GitError(f"'{local_path}' is not a valid Git repository")
        except GitCommandError as e:
            raise GitError(f"Failed to delete branch '{branch_name}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to delete branch '{branch_name}': {str(e)}")

    async def git_status(self, local_path: str) -> GitStatus:
        """Get Git repository status.
        
        Args:
            local_path: Local repository path
            
        Returns:
            GitStatus: Repository status information
        """
        try:
            repo = Repo(local_path)
            
            return GitStatus(
                is_dirty=repo.is_dirty(),
                untracked_files=repo.untracked_files,
                modified_files=[item.a_path for item in repo.index.diff(None)],
                staged_files=[item.a_path for item in repo.index.diff("HEAD")],
                current_branch=repo.active_branch.name,
                branches=[branch.name for branch in repo.branches]
            )
            
        except InvalidGitRepositoryError:
            raise GitError(f"'{local_path}' is not a valid Git repository")
        except Exception as e:
            raise GitError(f"Failed to get status: {str(e)}")
