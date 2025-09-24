"""GitHub Environment for AgentWorld - provides GitHub operations as an environment."""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel, Field, SecretStr, ConfigDict

from src.logger import logger
from src.environments.protocol.environment import BaseEnvironment
from src.environments.protocol import ecp
from src.environments.github import (
    GitHubService,
    AuthenticationError,
    NotFoundError,
    GitError,
    RepositoryError
)
from src.environments.github.types import (
    CreateRepositoryRequest,
    ForkRepositoryRequest,
    DeleteRepositoryRequest,
    GetRepositoryRequest, 
    CloneRepositoryRequest, 
    InitRepositoryRequest,
    GitCommitRequest, 
    GitPushRequest,
    GitPullRequest,
    GitFetchRequest,
    GitCreateBranchRequest,
    GitCheckoutBranchRequest,
    GitListBranchesRequest,
    GitDeleteBranchRequest,
    GitStatusRequest
)
from src.utils import dedent, get_env, assemble_project_path

@ecp.environment()
class GitHubEnvironment(BaseEnvironment):
    """GitHub Environment that provides GitHub operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="github", description="The name of the GitHub environment.")
    type: str = Field(default="GitHub", description="The type of the GitHub environment.")
    description: str = Field(default="GitHub environment for repository and Git operations", description="The description of the GitHub environment.")
    args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the GitHub environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the GitHub environment including repositories and Git status.",
        }
    }, description="The metadata of the GitHub environment.")
    
    def __init__(
        self,
        base_dir: str = None,
        token: Optional[SecretStr] = None,
        username: Optional[SecretStr] = None,
        **kwargs
    ):
        """
        Initialize the GitHub environment.
        
        Args:
            base_dir (str): Base directory for GitHub operations
            token (Optional[SecretStr]): GitHub Personal Access Token (PAT)
            username (Optional[SecretStr]): GitHub username
        """
        super().__init__(**kwargs)
        
        self.base_dir = assemble_project_path(base_dir)
        self.token = (token or get_env("GITHUB_TOKEN")).get_secret_value()
        self.username = (username or get_env("GITHUB_USERNAME")).get_secret_value()
        
        # Initialize GitHub service
        self.github_service = GitHubService(
            token=self.token,
            username=self.username
        )
        
    async def initialize(self) -> None:
        """Initialize the GitHub environment."""
        await self.github_service.initialize()
        logger.info(f"| 🚀 GitHub Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the GitHub environment."""
        await self.github_service.cleanup()
        logger.info("| 🧹 GitHub Environment cleanup completed")

    def _resolve_path(self, local_path: str) -> str:
        """Resolve local path relative to base directory.
        
        Args:
            local_path: The local path (can be relative or absolute)
            
        Returns:
            str: The resolved absolute path
        """
        path = Path(local_path)
        if path.is_absolute():
            return str(path.resolve())
        return str((self.base_dir / path).resolve())

    # --------------- Repository Operations ---------------
    @ecp.action(name="create_repository", 
                type="GitHub", 
                description="Create a new GitHub repository")
    async def create_repository(
        self,
        name: str,
        description: Optional[str] = None,
        private: bool = False,
        auto_init: bool = False
    ) -> str:
        """Create a new GitHub repository.
        
        Args:
            name (str): The name of the repository.
            description (Optional[str]): The description of the repository.
            private (bool): Whether the repository is private.
            auto_init (bool): Whether to auto-initialize the repository.
        
        Returns:
            A string indicating the success or failure of the repository creation.
        """
        try:
            # First check if repository already exists
            current_user = self.github_service.authenticated_user.login
            try:
                
                request = GetRepositoryRequest(owner=current_user, repo=name)
                existing_repo = await self.github_service.get_repository(request)
                
                if existing_repo.success and existing_repo.repository:
                    return f"Repository '{existing_repo.repository.full_name}' already exists. URL: {existing_repo.repository.html_url}"
            
            except RepositoryError:
                # Repository doesn't exist, proceed with creation
                pass
            
            request = CreateRepositoryRequest(
                name=name,
                description=description,
                private=private,
                auto_init=auto_init
            )
            result = await self.github_service.create_repository(request)
            return result.message
        
        except RepositoryError as e:
            return str(e)
        except Exception as e:
            return f"Failed to create repository '{name}': {str(e)}"

    @ecp.action(name="get_repository",
                type="GitHub", 
                description="Get your repository information")
    async def get_repository(self, repo: str) -> str:
        """Get repository information for your own repository.
        
        Args:
            repo (str): The name of your repository.
        
        Returns:
            A string containing detailed repository information including description, URL, stars, forks, language, privacy status, and timestamps.
        """
        try:
            # Use authenticated user as owner
            owner = self.github_service.authenticated_user.login
            request = GetRepositoryRequest(owner=owner, repo=repo)
            result = await self.github_service.get_repository(request)
            
            if not result.success or not result.repository:
                return f"| ❌ {result.message}"
            
            repository = result.repository
            
            result = dedent(f"""
                        Repository: {repository.full_name}
                        Description: {repository.description or 'No description'}
                        URL: {repository.html_url}
                        Stars: {repository.stargazers_count}
                        Forks: {repository.forks_count}
                        Language: {repository.language or 'Unknown'}
                        Private: {repository.private}
                        Created: {repository.created_at}
                        Updated: {repository.updated_at}
                        """)
            return result
        except NotFoundError as e:
            return str(e)
        except Exception as e:
            return f"Failed to get repository '{repo}': {str(e)}"

    @ecp.action(name="fork_repository",
                type="GitHub", 
                description="Fork a public repository to your account")
    async def fork_repository(self, owner: str, repo: str) -> str:
        """Fork a public repository to your account.
        
        Args:
            owner (str): The owner of the repository to fork.
            repo (str): The name of the repository to fork.
        
        Returns:
            A string indicating the success or failure of the fork operation.
        """
        try:
            request = ForkRepositoryRequest(owner=owner, repo=repo)
            result = await self.github_service.fork_repository(request)
            return result.message
        except RepositoryError as e:
            return str(e)
        except Exception as e:
            return f"Failed to fork repository '{owner}/{repo}': {str(e)}"

    @ecp.action(name="delete_repository",
                type="GitHub", 
                description="Delete your own repository")
    async def delete_repository(self, repo: str) -> str:
        """Delete your own repository.
        
        Args:
            repo (str): The name of your repository to delete.
        
        Returns:
            A string indicating the success or failure of the repository deletion operation.
        """
        try:
            # Use authenticated user as owner
            owner = self.github_service.authenticated_user.login
            request = DeleteRepositoryRequest(owner=owner, repo=repo)
            result = await self.github_service.delete_repository(request)
            return result.message
        except RepositoryError as e:
            return str(e)
        except Exception as e:
            return f"Failed to delete repository '{repo}': {str(e)}"

    # --------------- Git Operations ---------------
    @ecp.action(name="git_init", 
                type="GitHub", 
                description="Initialize a local directory as Git repository")
    async def git_init(
        self,
        local_path: str,
        remote_url: Optional[str] = None
    ) -> str:
        """Initialize a local directory as Git repository.
        
        Args:
            local_path (str): The local directory path to initialize as a Git repository (relative to base_dir).
            remote_url (Optional[str]): The remote repository URL to set as origin. If None, no remote is added.
        
        Returns:
            A string indicating the success or failure of the repository initialization operation.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = InitRepositoryRequest(local_path=resolved_path, remote_url=remote_url)
            result = await self.github_service.init_repository(request)
            return result.message
        except (GitError, RepositoryError) as e:
            return str(e)
        except Exception as e:
            return f"Failed to initialize repository in '{local_path}': {str(e)}"
    
    @ecp.action(name="git_clone", 
                type="GitHub", 
                description="Clone a repository to local directory (automatically forks if not your repository)")
    async def git_clone(
        self,
        owner: str,
        repo: str,
        local_path: str,
        branch: Optional[str] = None
    ) -> str:
        """Clone a repository to local directory.
        
        If the repository belongs to someone else, it will automatically fork the repository
        to your account first, then clone the forked version.
        
        Args:
            owner (str): The owner of the repository.
            repo (str): The name of the repository.
            local_path (str): The local directory path where the repository will be cloned (relative to base_dir).
            branch (Optional[str]): The specific branch to clone. If None, clones the default branch.
        
        Returns:
            A string indicating the success or failure of the repository cloning operation.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            current_user = self.github_service.authenticated_user.login
            
            # If it's the user's own repository, clone directly
            if owner == current_user:
                request = CloneRepositoryRequest(
                    owner=owner,
                    repo=repo,
                    local_path=resolved_path,
                    branch=branch
                )
                result = await self.github_service.clone_repository(request)
                return result.message
            
            # If it's someone else's repository, fork it first
            try:
                fork_request = ForkRepositoryRequest(owner=owner, repo=repo)
                fork_result = await self.github_service.fork_repository(fork_request)
                
                if not fork_result.success or not fork_result.repository:
                    return f"| ❌ Failed to fork repository: {fork_result.message}"
                
                fork_msg = fork_result.message
                
                # Clone the forked repository
                clone_request = CloneRepositoryRequest(
                    owner=fork_result.repository.owner,
                    repo=fork_result.repository.name,
                    local_path=resolved_path,
                    branch=branch
                )
                clone_result = await self.github_service.clone_repository(clone_request)
                
                return fork_msg + clone_result.message
                
            except RepositoryError as e:
                # If fork fails (e.g., already forked), try to clone the existing fork
                if "already be forked" in str(e).lower():
                    # Try to clone the user's existing fork
                    try:
                        clone_request = CloneRepositoryRequest(
                            owner=current_user,
                            repo=repo,
                            local_path=resolved_path,
                            branch=branch
                        )
                        clone_result = await self.github_service.clone_repository(clone_request)
                        return clone_result.message
                    except Exception:
                        pass
                raise e
                
        except (GitError, RepositoryError) as e:
            return str(e)
        except Exception as e:
            return f"Failed to clone repository '{owner}/{repo}': {str(e)}"
    
    @ecp.action(name="git_commit",
                type="GitHub", 
                description="Commit changes to local repository")
    async def git_commit(
        self,
        local_path: str,
        message: str,
        add_all: bool = True
    ) -> str:
        """Commit changes to local repository.
        
        Args:
            local_path (str): The local repository path (relative to base_dir).
            message (str): The commit message.
            add_all (bool): Whether to add all changes before committing. Defaults to True.
        
        Returns:
            A string indicating the success or failure of the commit operation, including the commit hash.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitCommitRequest(
                local_path=resolved_path,
                message=message,
                files=None if add_all else []
            )
            result = await self.github_service.git_commit(request)
            return result.message
        except GitError as e:
            return str(e)
        except Exception as e:
            return f"Failed to commit in '{local_path}': {str(e)}"

    @ecp.action(name="git_push",
                type="GitHub", 
                description="Push changes to remote repository")
    async def git_push(
        self,
        local_path: str,
        remote: str = "origin",
        branch: Optional[str] = None
    ) -> str:
        """Push changes to remote repository.
        
        Args:
            local_path (str): The local repository path (relative to base_dir).
            remote (str): The remote name. Defaults to "origin".
            branch (Optional[str]): The branch name to push. If None, pushes the current branch.
        
        Returns:
            A string indicating the success or failure of the push operation.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitPushRequest(
                local_path=resolved_path,
                remote=remote,
                branch=branch
            )
            result = await self.github_service.git_push(request)
            return result.message
        except GitError as e:
            return str(e)
        except Exception as e:
            return f"Failed to push from '{local_path}': {str(e)}"

    @ecp.action(name="git_pull",
                type="GitHub", 
                description="Pull changes from remote repository")
    async def git_pull(
        self,
        local_path: str,
        remote: str = "origin",
        branch: Optional[str] = None
    ) -> str:
        """Pull changes from remote repository.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
            remote (str): The remote name. Defaults to "origin".
            branch (Optional[str]): The branch name to pull. If None, pulls the current branch.
        
        Returns:
            A string indicating the success or failure of the pull operation.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitPullRequest(
                local_path=resolved_path,
                remote=remote,
                branch=branch
            )
            result = await self.github_service.git_pull(request)
            return result.message
        except GitError as e:
            return str(e)
        except Exception as e:
            return f"Failed to pull to '{local_path}': {str(e)}"

    @ecp.action(name="git_fetch",
                type="GitHub", 
                description="Fetch changes from remote repository")
    async def git_fetch(
        self,
        local_path: str,
        remote: str = "origin"
    ) -> str:
        """Fetch changes from remote repository.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
            remote (str): The remote name. Defaults to "origin".
        
        Returns:
            A string indicating the success or failure of the fetch operation.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitFetchRequest(
                local_path=resolved_path,
                remote=remote
            )
            result = await self.github_service.git_fetch(request)
            return result.message
        except GitError as e:
            return str(e)
        except Exception as e:
            return f"Failed to fetch to '{local_path}': {str(e)}"

    # --------------- Branch Operations ---------------
    @ecp.action(name="git_create_branch",
                type="GitHub", 
                description="Create a new branch")
    async def git_create_branch(
        self,
        local_path: str,
        branch_name: str,
        checkout: bool = True
    ) -> str:
        """Create a new branch.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
            branch_name (str): The name of the new branch to create.
            checkout (bool): Whether to checkout the new branch after creation. Defaults to True.
        
        Returns:
            A string indicating the success or failure of the branch creation operation.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitCreateBranchRequest(
                local_path=resolved_path,
                branch_name=branch_name,
                from_branch=None
            )
            result = await self.github_service.git_create_branch(request)
            return result.message
        except GitError as e:
            return str(e)
        except Exception as e:
            return f"Failed to create branch '{branch_name}': {str(e)}"

    @ecp.action(name="git_checkout_branch",
                type="GitHub", 
                description="Checkout an existing branch")
    async def git_checkout_branch(
        self,
        local_path: str,
        branch_name: str
    ) -> str:
        """Checkout an existing branch.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
            branch_name (str): The name of the branch to checkout.
        
        Returns:
            A string indicating the success or failure of the branch checkout operation.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitCheckoutBranchRequest(
                local_path=resolved_path,
                branch_name=branch_name
            )
            result = await self.github_service.git_checkout_branch(request)
            return result.message
        except GitError as e:
            return str(e)
        except Exception as e:
            return f"Failed to checkout branch '{branch_name}': {str(e)}"

    @ecp.action(name="git_list_branches",
                type="GitHub", 
                description="List all branches")
    async def git_list_branches(self, local_path: str) -> str:
        """List all branches.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
        
        Returns:
            A string containing a formatted list of all branches with the current branch marked.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitListBranchesRequest(local_path=resolved_path)
            result = await self.github_service.git_list_branches(request)
            return result.message
        except GitError as e:
            return str(e)
        except Exception as e:
            return f"Failed to list branches: {str(e)}"

    @ecp.action(name="git_delete_branch",
                type="GitHub", 
                description="Delete a branch")
    async def git_delete_branch(
        self,
        local_path: str,
        branch_name: str,
        force: bool = False
    ) -> str:
        """Delete a branch.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
            branch_name (str): The name of the branch to delete.
            force (bool): Whether to force delete the branch even if it's not merged. Defaults to False.
        
        Returns:
            A string indicating the success or failure of the branch deletion operation.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitDeleteBranchRequest(
                local_path=resolved_path,
                branch_name=branch_name,
                force=force
            )
            result = await self.github_service.git_delete_branch(request)
            return result.message
        except GitError as e:
            return str(e)
        except Exception as e:
            return f"Failed to delete branch '{branch_name}': {str(e)}"

    @ecp.action(name="git_status",
                type="GitHub", 
                description="Get Git repository status")
    async def git_status(self, local_path: str) -> str:
        """Get Git repository status.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
        
        Returns:
            A string containing detailed repository status including current branch, dirty state, modified files, staged files, untracked files, and all branches.
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitStatusRequest(local_path=resolved_path)
            result = await self.github_service.git_status(request)
            
            if not result.success or not result.status:
                return f"| ❌ {result.message}"
            
            status = result.status
            status_text = dedent(f"""Repository Status:
                Current Branch: {status.current_branch}
                Dirty: {status.is_dirty}
                Modified Files: {', '.join(status.modified_files) if status.modified_files else 'None'}
                Staged Files: {', '.join(status.staged_files) if status.staged_files else 'None'}
                Untracked Files: {', '.join(status.untracked_files) if status.untracked_files else 'None'}
                Branches: {', '.join(status.branches)}""")
            return status_text
        except GitError as e:
            return str(e)
        except Exception as e:
            return f"Failed to get status: {str(e)}"

    # --------------- Environment Interface Methods ---------------
    async def get_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "type": "github",
            "username": self.github_service.authenticated_user.login if self.github_service else None,
            "authenticated": self.github_service is not None,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if self.github_service is None:
                return {"status": "unhealthy", "error": "Not initialized"}
            
            # Test service access
            user = self.github_service.authenticated_user
            return {
                "status": "healthy",
                "username": user.login,
                "authenticated": True,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the GitHub environment."""
        try:
            git_status = await self.git_status(self.base_dir)
            
            state = dedent(f"""
                GitHub Environment:
                Username: {self.username}
                Authenticated: {bool(self.token)}
                Service Available: {self.github_service is not None}
                Git Status: {git_status}
            """)
            
            return {
                "state": state
            }
        except Exception as e:
            logger.error(f"Failed to get GitHub state: {e}")
            return {
                "state": str(e)
            }