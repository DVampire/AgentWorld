"""GitHub service implementation using PyGithub + GitPython."""
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

from github import Github, GithubException
from git import Repo, InvalidGitRepositoryError, GitCommandError

from src.environments.github.types import GitHubRepository, GitHubUser, GitHubBranch, GitStatus
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

    async def create_repository(
        self,
        name: str,
        description: Optional[str] = None,
        private: bool = False,
        auto_init: bool = False
    ) -> GitHubRepository:
        """Create a new GitHub repository.
        
        Args:
            name: Repository name
            description: Repository description
            private: Whether repository is private
            auto_init: Whether to initialize with README
            
        Returns:
            GitHubRepository: Created repository information
        """
        try:
            # Get the PyGithub user object for creating repositories
            github_user = await asyncio.to_thread(self.github.get_user)
            # Use asyncio.to_thread to run the synchronous GitHub API call in a thread pool
            repo = await asyncio.to_thread(
                github_user.create_repo,
                name=name,
                description=description,
                private=private,
                auto_init=auto_init
            )
            
            return GitHubRepository(
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
            
        except GithubException as e:
            if e.status == 401:
                raise AuthenticationError(f"Invalid GitHub token or insufficient permissions")
            elif e.status == 403:
                raise RepositoryError(f"Permission denied: Cannot create repository '{name}'")
            elif e.status == 422:
                raise RepositoryError(f"Repository '{name}' already exists or invalid name")
            elif e.status == 404:
                raise NotFoundError(f"User not found or repository creation failed")
            else:
                raise RepositoryError(f"Failed to create repository '{name}': {e}")
        except Exception as e:
            raise RepositoryError(f"Failed to create repository '{name}': {e}")

    async def fork_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Fork a repository to your account.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            GitHubRepository: Forked repository information
        """
        try:
            # Get the original repository
            original_repo = await asyncio.to_thread(self.github.get_repo, f"{owner}/{repo}")
            
            # Fork the repository
            github_user = await asyncio.to_thread(self.github.get_user)
            forked_repo = await asyncio.to_thread(github_user.create_fork, original_repo)
            
            return GitHubRepository(
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
            
        except GithubException as e:
            if e.status == 422:
                raise RepositoryError(f"Repository '{owner}/{repo}' cannot be forked (may already be forked or private)")
            elif e.status == 404:
                raise NotFoundError(f"Repository '{owner}/{repo}' not found")
            raise RepositoryError(f"Failed to fork repository '{owner}/{repo}': {e}")
        except Exception as e:
            raise RepositoryError(f"Failed to fork repository '{owner}/{repo}': {e}")

    async def delete_repository(self, owner: str, repo: str) -> None:
        """Delete a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Raises:
            RepositoryError: If deletion fails
        """
        try:
            repository = await asyncio.to_thread(self.github.get_repo, f"{owner}/{repo}")
            await asyncio.to_thread(repository.delete)
            
        except GithubException as e:
            if e.status == 404:
                raise NotFoundError(f"Repository '{owner}/{repo}' not found")
            elif e.status == 403:
                raise RepositoryError(f"Permission denied: Cannot delete repository '{owner}/{repo}'")
            raise RepositoryError(f"Failed to delete repository '{owner}/{repo}': {e}")
        except Exception as e:
            raise RepositoryError(f"Failed to delete repository '{owner}/{repo}': {e}")

    async def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Get repository information.
        
        Args:
            owner: Repository owner
            repo: Repository name
            
        Returns:
            GitHubRepository: Repository information
        """
        try:
            
            repository = await asyncio.to_thread(self.github.get_repo, f"{owner}/{repo}")
            
            return GitHubRepository(
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
            
        except GithubException as e:
            if e.status == 404:
                raise NotFoundError(f"Repository '{owner}/{repo}' not found")
            raise GitHubError(f"Failed to get repository '{owner}/{repo}': {e}")
        except Exception as e:
            raise GitHubError(f"Failed to get repository '{owner}/{repo}': {e}")

    async def clone_repository(
        self,
        owner: str,
        repo: str,
        local_path: str,
        branch: Optional[str] = None
    ) -> str:
        """Clone a repository to local directory.
        
        Args:
            owner: Repository owner
            repo: Repository name
            local_path: Local directory path to clone to
            branch: Specific branch to clone (optional)
            
        Returns:
            str: Clone result message
        """
        try:
            repo_url = f"https://github.com/{owner}/{repo}.git"
            local_path = Path(local_path)
            
            if local_path.exists():
                raise RepositoryError(f"Directory '{local_path}' already exists")
            
            # Clone repository
            if branch:
                Repo.clone_from(repo_url, local_path, branch=branch)
                return f"Repository '{owner}/{repo}' cloned to '{local_path}' (branch: {branch})"
            else:
                Repo.clone_from(repo_url, local_path)
                return f"Repository '{owner}/{repo}' cloned to '{local_path}'"
                
        except GitCommandError as e:
            raise GitError(f"Failed to clone repository '{owner}/{repo}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to clone repository '{owner}/{repo}': {str(e)}")

    async def init_repository(
        self,
        local_path: str,
        remote_url: Optional[str] = None
    ) -> str:
        """Initialize a local directory as Git repository.
        
        Args:
            local_path: Local directory path
            remote_url: Remote repository URL (optional)
            
        Returns:
            str: Initialization result message
        """
        try:
            local_path = Path(local_path)
            
            if not local_path.exists():
                raise RepositoryError(f"Directory '{local_path}' does not exist")
            
            if (local_path / '.git').exists():
                raise RepositoryError(f"Directory '{local_path}' is already a Git repository")
            
            # Initialize repository
            repo = Repo.init(local_path)
            
            result = f"Git repository initialized in '{local_path}'"
            
            if remote_url:
                try:
                    repo.create_remote('origin', remote_url)
                    result += f" with remote origin: {remote_url}"
                except Exception as e:
                    result += f" (failed to add remote: {str(e)})"
            
            return result
            
        except Exception as e:
            raise GitError(f"Failed to initialize repository in '{local_path}': {str(e)}")

    # --------------- Git Operations ---------------

    async def git_commit(
        self,
        local_path: str,
        message: str,
        add_all: bool = True
    ) -> str:
        """Commit changes to local repository.
        
        Args:
            local_path: Local repository path
            message: Commit message
            add_all: Whether to add all changes
            
        Returns:
            str: Commit result message
        """
        try:
            repo = Repo(local_path)
            
            if add_all:
                repo.git.add(A=True)
            
            # Check if there are changes to commit
            if not repo.is_dirty() and not repo.untracked_files:
                return "No changes to commit"
            
            # Commit changes
            commit = repo.index.commit(message)
            return f"Commit created: {commit.hexsha[:8]} - {message}"
            
        except InvalidGitRepositoryError:
            raise GitError(f"'{local_path}' is not a valid Git repository")
        except GitCommandError as e:
            raise GitError(f"Failed to commit in '{local_path}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to commit in '{local_path}': {str(e)}")

    async def git_push(
        self,
        local_path: str,
        remote: str = "origin",
        branch: Optional[str] = None
    ) -> str:
        """Push changes to remote repository.
        
        Args:
            local_path: Local repository path
            remote: Remote name (default: origin)
            branch: Branch name (optional, uses current branch if not specified)
            
        Returns:
            str: Push result message
        """
        try:
            repo = Repo(local_path)
            
            if branch is None:
                branch = repo.active_branch.name
            
            # Push to remote
            origin = repo.remote(remote)
            origin.push(branch)
            
            return f"Successfully pushed branch '{branch}' to remote '{remote}'"
            
        except InvalidGitRepositoryError:
            raise GitError(f"'{local_path}' is not a valid Git repository")
        except GitCommandError as e:
            raise GitError(f"Failed to push from '{local_path}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to push from '{local_path}': {str(e)}")

    async def git_pull(
        self,
        local_path: str,
        remote: str = "origin",
        branch: Optional[str] = None
    ) -> str:
        """Pull changes from remote repository.
        
        Args:
            local_path: Local repository path
            remote: Remote name (default: origin)
            branch: Branch name (optional, uses current branch if not specified)
            
        Returns:
            str: Pull result message
        """
        try:
            repo = Repo(local_path)
            
            if branch is None:
                branch = repo.active_branch.name
            
            # Pull from remote
            origin = repo.remote(remote)
            origin.pull(branch)
            
            return f"Successfully pulled branch '{branch}' from remote '{remote}'"
            
        except InvalidGitRepositoryError:
            raise GitError(f"'{local_path}' is not a valid Git repository")
        except GitCommandError as e:
            raise GitError(f"Failed to pull to '{local_path}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to pull to '{local_path}': {str(e)}")

    async def git_fetch(
        self,
        local_path: str,
        remote: str = "origin"
    ) -> str:
        """Fetch changes from remote repository.
        
        Args:
            local_path: Local repository path
            remote: Remote name (default: origin)
            
        Returns:
            str: Fetch result message
        """
        try:
            repo = Repo(local_path)
            
            # Fetch from remote
            origin = repo.remote(remote)
            origin.fetch()
            
            return f"Successfully fetched from remote '{remote}'"
            
        except InvalidGitRepositoryError:
            raise GitError(f"'{local_path}' is not a valid Git repository")
        except GitCommandError as e:
            raise GitError(f"Failed to fetch to '{local_path}': {str(e)}")
        except Exception as e:
            raise GitError(f"Failed to fetch to '{local_path}': {str(e)}")

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
