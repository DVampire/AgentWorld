"""GitHub service implementation using PyGithub + GitPython."""
import asyncio
from pathlib import Path
from typing import Optional, List, Tuple

from github import Github, GithubException
from git import Repo, InvalidGitRepositoryError, GitCommandError

from src.environments.github.types import (
    GitHubRepository,
    GitHubUser, 
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
                message=f"Successfully forked repository from {request.owner}/{request.repo} to {repository.full_name}/{repository.name}"
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

    def _execute_git_command(self, args: List[str], cwd: str = None) -> Tuple[bool, str, str]:
        """Execute a git command and return (success, stdout, stderr)."""
        import subprocess
        try:
            result = subprocess.run(
                ['git'] + args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Git command timed out"
        except Exception as e:
            return False, "", str(e)

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
            
            # Clone repository using git command
            if request.branch:
                success, stdout, stderr = self._execute_git_command([
                    'clone', '-b', request.branch, repo_url, str(local_path)
                ])
            else:
                success, stdout, stderr = self._execute_git_command([
                    'clone', repo_url, str(local_path)
                ])
            
            if not success:
                return CloneRepositoryResult(
                    local_path=request.local_path,
                    success=False,
                    message=f"Failed to clone repository: {stderr}"
                )
            
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
            success, stdout, stderr = self._execute_git_command(['init'], cwd=request.local_path)
            if not success:
                return InitRepositoryResult(
                    local_path=request.local_path,
                    success=False,
                    message=f"Failed to initialize repository: {stderr}"
                )
            
            message = f"Git repository initialized in '{request.local_path}'"
            
            if request.remote_url:
                success, stdout, stderr = self._execute_git_command(['remote', 'add', 'origin', request.remote_url], cwd=request.local_path)
                if success:
                    message += f" with remote origin: {request.remote_url}"
                else:
                    message += f" (failed to add remote: {stderr})"
            
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
            # Add files using git command
            if request.files is None:  # add_all
                success, stdout, stderr = self._execute_git_command(['add', '-A'], cwd=request.local_path)
            elif request.files:
                success, stdout, stderr = self._execute_git_command(['add'] + request.files, cwd=request.local_path)
            else:
                success, stdout, stderr = True, "", ""
            
            if not success:
                return GitCommitResult(
                    local_path=request.local_path,
                    commit_hash=None,
                    success=False,
                    message=f"Failed to add files: {stderr}"
                )
            
            # Check if there are changes to commit
            success, stdout, stderr = self._execute_git_command(['diff', '--cached', '--quiet'], cwd=request.local_path)
            if success:  # No changes to commit
                return GitCommitResult(
                    local_path=request.local_path,
                    commit_hash=None,
                    success=False,
                    message="No changes to commit"
                )
            
            # Commit changes
            success, stdout, stderr = self._execute_git_command(['commit', '-m', request.message], cwd=request.local_path)
            if not success:
                return GitCommitResult(
                    local_path=request.local_path,
                    commit_hash=None,
                    success=False,
                    message=f"Failed to commit: {stderr}"
                )
            
            # Get commit hash
            success, commit_hash, stderr = self._execute_git_command(['rev-parse', 'HEAD'], cwd=request.local_path)
            commit_hash = commit_hash.strip() if success else "unknown"
            
            return GitCommitResult(
                local_path=request.local_path,
                commit_hash=commit_hash,
                success=True,
                message=f"Commit created: {commit_hash[:8]} - {request.message}"
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
            # Get current branch if not specified
            if request.branch is None:
                success, branch, stderr = self._execute_git_command(['branch', '--show-current'], cwd=request.local_path)
                if not success:
                    return GitPushResult(
                        local_path=request.local_path,
                        success=False,
                        message=f"Failed to get current branch: {stderr}"
                    )
                branch = branch.strip()
            else:
                branch = request.branch
            
            # Check if remote exists
            success, stdout, stderr = self._execute_git_command(['remote', 'get-url', request.remote], cwd=request.local_path)
            if not success:
                return GitPushResult(
                    local_path=request.local_path,
                    success=False,
                    message=f"Remote named '{request.remote}' didn't exist"
                )
            
            # Push to remote
            success, stdout, stderr = self._execute_git_command(['push', request.remote, branch], cwd=request.local_path)
            if not success:
                return GitPushResult(
                    local_path=request.local_path,
                    success=False,
                    message=f"Failed to push: {stderr}"
                )
            
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
            # Get current branch if not specified
            if request.branch is None:
                success, branch, stderr = self._execute_git_command(['branch', '--show-current'], cwd=request.local_path)
                if not success:
                    return GitPullResult(
                        local_path=request.local_path,
                        success=False,
                        message=f"Failed to get current branch: {stderr}"
                    )
                branch = branch.strip()
            else:
                branch = request.branch
            
            # Pull from remote
            success, stdout, stderr = self._execute_git_command(['pull', request.remote, branch], cwd=request.local_path)
            if not success:
                return GitPullResult(
                    local_path=request.local_path,
                    success=False,
                    message=f"Failed to pull: {stderr}"
                )
            
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
            # Fetch from remote
            success, stdout, stderr = self._execute_git_command(['fetch', request.remote], cwd=request.local_path)
            if not success:
                return GitFetchResult(
                    local_path=request.local_path,
                    success=False,
                    message=f"Failed to fetch: {stderr}"
                )
            
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

    async def git_create_branch(self, request: GitCreateBranchRequest) -> GitCreateBranchResult:
        """Create a new branch.
        
        Args:
            request: GitCreateBranchRequest with local_path, branch_name, from_branch
            
        Returns:
            GitCreateBranchResult: Branch creation result
        """
        try:
            # Check if branch already exists
            success, stdout, stderr = self._execute_git_command(['branch', '--list', request.branch_name], cwd=request.local_path)
            if success and stdout.strip():
                return GitCreateBranchResult(
                    local_path=request.local_path,
                    branch_name=request.branch_name,
                    success=False,
                    message=f"Branch '{request.branch_name}' already exists"
                )
            
            # Create new branch
            if request.from_branch:
                success, stdout, stderr = self._execute_git_command(['checkout', '-b', request.branch_name, request.from_branch], cwd=request.local_path)
            else:
                success, stdout, stderr = self._execute_git_command(['checkout', '-b', request.branch_name], cwd=request.local_path)
            
            if not success:
                return GitCreateBranchResult(
                    local_path=request.local_path,
                    branch_name=request.branch_name,
                    success=False,
                    message=f"Failed to create and checkout branch '{request.branch_name}': {stderr}"
                )
            
            return GitCreateBranchResult(
                local_path=request.local_path,
                branch_name=request.branch_name,
                success=True,
                message=f"Branch '{request.branch_name}' created and checked out"
            )
                
        except InvalidGitRepositoryError:
            raise GitError(f"'{request.local_path}' is not a valid Git repository")
        except GitCommandError as e:
            raise GitError(f"Failed to create branch '{request.branch_name}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to create branch '{request.branch_name}': {str(e)}")

    async def git_checkout_branch(self, request: GitCheckoutBranchRequest) -> GitCheckoutBranchResult:
        """Checkout an existing branch.
        
        Args:
            request: GitCheckoutBranchRequest with local_path, branch_name
            
        Returns:
            GitCheckoutBranchResult: Checkout result
        """
        try:
            # Checkout branch
            success, stdout, stderr = self._execute_git_command(['checkout', request.branch_name], cwd=request.local_path)
            if not success:
                return GitCheckoutBranchResult(
                    local_path=request.local_path,
                    branch_name=request.branch_name,
                    success=False,
                    message=f"Failed to checkout branch '{request.branch_name}': {stderr}"
                )
            
            return GitCheckoutBranchResult(
                local_path=request.local_path,
                branch_name=request.branch_name,
                success=True,
                message=f"Checked out branch '{request.branch_name}'"
            )
            
        except InvalidGitRepositoryError:
            raise GitError(f"'{request.local_path}' is not a valid Git repository")
        except GitCommandError as e:
            raise GitError(f"Failed to checkout branch '{request.branch_name}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to checkout branch '{request.branch_name}': {str(e)}")

    async def git_list_branches(self, request: GitListBranchesRequest) -> GitListBranchesResult:
        """List all branches.
        
        Args:
            request: GitListBranchesRequest with local_path
            
        Returns:
            GitListBranchesResult: List of branches
        """
        try:
            # Get current branch
            success, current_branch, stderr = self._execute_git_command(['branch', '--show-current'], cwd=request.local_path)
            if not success:
                return GitListBranchesResult(
                    local_path=request.local_path,
                    branches=[],
                    current_branch=None,
                    success=False,
                    message=f"Failed to get current branch: {stderr}"
                )
            current_branch = current_branch.strip()
            
            # Get all branches
            success, stdout, stderr = self._execute_git_command(['branch', '-a'], cwd=request.local_path)
            if not success:
                return GitListBranchesResult(
                    local_path=request.local_path,
                    branches=[],
                    current_branch=current_branch,
                    success=False,
                    message=f"Failed to list branches: {stderr}"
                )
            
            branches = []
            for line in stdout.split('\n'):
                line = line.strip()
                if line:
                    # Remove * and remote/ prefix
                    branch_name = line.replace('*', '').strip()
                    if branch_name.startswith('remotes/'):
                        continue  # Skip remote branches for now
                    
                    branches.append(branch_name)
            
            return GitListBranchesResult(
                local_path=request.local_path,
                branches=branches,
                current_branch=current_branch,
                success=True,
                message=f"Found {len(branches)} branches"
            )
            
        except InvalidGitRepositoryError:
            raise GitError(f"'{request.local_path}' is not a valid Git repository")
        except Exception as e:
            raise GitError(f"Failed to list branches: {str(e)}")

    async def git_delete_branch(self, request: GitDeleteBranchRequest) -> GitDeleteBranchResult:
        """Delete a branch.
        
        Args:
            request: GitDeleteBranchRequest with local_path, branch_name, force
            
        Returns:
            GitDeleteBranchResult: Delete result
        """
        try:
            # Get current branch
            success, current_branch, stderr = self._execute_git_command(['branch', '--show-current'], cwd=request.local_path)
            if not success:
                return GitDeleteBranchResult(
                    local_path=request.local_path,
                    branch_name=request.branch_name,
                    success=False,
                    message=f"Failed to get current branch: {stderr}"
                )
            current_branch = current_branch.strip()
            
            # Check if trying to delete current branch
            if request.branch_name == current_branch:
                return GitDeleteBranchResult(
                    local_path=request.local_path,
                    branch_name=request.branch_name,
                    success=False,
                    message=f"Cannot delete current branch '{request.branch_name}'. Switch to another branch first."
                )
            
            # Delete branch
            if request.force:
                success, stdout, stderr = self._execute_git_command(['branch', '-D', request.branch_name], cwd=request.local_path)
                if not success:
                    return GitDeleteBranchResult(
                        local_path=request.local_path,
                        branch_name=request.branch_name,
                        success=False,
                        message=f"Failed to force delete branch '{request.branch_name}': {stderr}"
                    )
                return GitDeleteBranchResult(
                    local_path=request.local_path,
                    branch_name=request.branch_name,
                    success=True,
                    message=f"Branch '{request.branch_name}' force deleted"
                )
            else:
                success, stdout, stderr = self._execute_git_command(['branch', '-d', request.branch_name], cwd=request.local_path)
                if not success:
                    return GitDeleteBranchResult(
                        local_path=request.local_path,
                        branch_name=request.branch_name,
                        success=False,
                        message=f"Failed to delete branch '{request.branch_name}': {stderr}"
                    )
                return GitDeleteBranchResult(
                    local_path=request.local_path,
                    branch_name=request.branch_name,
                    success=True,
                    message=f"Branch '{request.branch_name}' deleted"
                )
                
        except InvalidGitRepositoryError:
            raise GitError(f"'{request.local_path}' is not a valid Git repository")
        except GitCommandError as e:
            raise GitError(f"Failed to delete branch '{request.branch_name}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to delete branch '{request.branch_name}': {str(e)}")

    async def git_status(self, local_path: str) -> GitStatus:
        """Get Git repository status.
        
        Args:
            local_path: Local repository path
            
        Returns:
            GitStatus: Repository status information
        """
        try:
            # Get current branch
            success, current_branch, stderr = self._execute_git_command(['branch', '--show-current'], cwd=local_path)
            if not success:
                raise GitError(f"Failed to get current branch: {stderr}")
            current_branch = current_branch.strip()
            
            # Get all branches
            success, stdout, stderr = self._execute_git_command(['branch'], cwd=local_path)
            if not success:
                raise GitError(f"Failed to get branches: {stderr}")
            
            branches = []
            for line in stdout.split('\n'):
                line = line.strip()
                if line:
                    branch_name = line.replace('*', '').strip()
                    branches.append(branch_name)
            
            # Get status information
            success, stdout, stderr = self._execute_git_command(['status', '--porcelain'], cwd=local_path)
            if not success:
                raise GitError(f"Failed to get status: {stderr}")
            
            modified_files = []
            staged_files = []
            untracked_files = []
            
            for line in stdout.split('\n'):
                line = line.strip()
                if line:
                    status = line[:2]
                    filename = line[3:]
                    
                    if status.startswith('M') or status.startswith('A') or status.startswith('D'):
                        if status[0] != ' ':
                            staged_files.append(filename)
                        if status[1] != ' ':
                            modified_files.append(filename)
                    elif status.startswith('??'):
                        untracked_files.append(filename)
            
            # Check if repository is dirty
            is_dirty = len(modified_files) > 0 or len(untracked_files) > 0 or len(staged_files) > 0
            
            return GitStatus(
                is_dirty=is_dirty,
                untracked_files=untracked_files,
                modified_files=modified_files,
                staged_files=staged_files,
                current_branch=current_branch,
                branches=branches
            )
            
        except InvalidGitRepositoryError:
            raise GitError(f"'{local_path}' is not a valid Git repository")
        except Exception as e:
            raise GitError(f"Failed to get status: {str(e)}")
