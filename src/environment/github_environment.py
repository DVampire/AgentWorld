"""GitHub Environment for AgentWorld - provides GitHub operations as an environment."""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
from pydantic import BaseModel, Field, SecretStr, ConfigDict

from src.logger import logger
from src.environment.protocol.environment import BaseEnvironment
from src.environment.protocol import ecp
from src.environment.github import (
    GitHubService,
    AuthenticationError,
    NotFoundError,
    GitError,
    RepositoryError
)
from src.environment.github.types import (
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
            "interaction": dedent(f"""
                Guidelines for interacting with the GitHub environment:
                - If DO NOT have remote URL, you should `create_repository` first. Then you can `git_clone` the repository.
            """),
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
    ) -> Dict[str, Any]:
        """Create a new GitHub repository.
        
        Args:
            name (str): The name of the repository.
            description (Optional[str]): The description of the repository.
            private (bool): Whether the repository is private.
            auto_init (bool): Whether to auto-initialize the repository.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # First check if repository already exists
            current_user = self.github_service.authenticated_user.login
            try:
                request = GetRepositoryRequest(owner=current_user, repo=name)
                existing_repo = await self.github_service.get_repository(request)
                
                if existing_repo.success and "repository" in existing_repo.extra:
                    repo_dict = existing_repo.extra["repository"]
                    return {
                        "success": True,
                        "message": f"Repository '{repo_dict.get('full_name', name)}' already exists. URL: {repo_dict.get('html_url', '')}",
                        "extra": existing_repo.extra
                    }
            
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
            
            extra = result.extra.copy() if result.extra else {}
            extra["name"] = name
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        
        except RepositoryError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "name": name}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create repository '{name}': {str(e)}",
                "extra": {"error": str(e), "name": name}
            }

    @ecp.action(name="get_repository",
                type="GitHub", 
                description="Get your repository information")
    async def get_repository(self, repo: str) -> Dict[str, Any]:
        """Get repository information for your own repository.
        
        Args:
            repo (str): The name of your repository.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # Use authenticated user as owner
            owner = self.github_service.authenticated_user.login
            request = GetRepositoryRequest(owner=owner, repo=repo)
            result = await self.github_service.get_repository(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["repo"] = repo
            extra["owner"] = owner
            
            if result.success and "repository" in extra:
                repository = extra["repository"]
                message = dedent(f"""
                    Repository: {repository.get('full_name', repo)}
                    Description: {repository.get('description') or 'No description'}
                    URL: {repository.get('html_url', '')}
                    Stars: {repository.get('stargazers_count', 0)}
                    Forks: {repository.get('forks_count', 0)}
                    Language: {repository.get('language') or 'Unknown'}
                    Private: {repository.get('private', False)}
                    Created: {repository.get('created_at', 'Unknown')}
                    Updated: {repository.get('updated_at', 'Unknown')}
                    """)
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except NotFoundError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "repo": repo}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get repository '{repo}': {str(e)}",
                "extra": {"error": str(e), "repo": repo}
            }

    @ecp.action(name="fork_repository",
                type="GitHub", 
                description="Fork a public repository to your account")
    async def fork_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """Fork a public repository to your account.
        
        Args:
            owner (str): The owner of the repository to fork.
            repo (str): The name of the repository to fork.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = ForkRepositoryRequest(owner=owner, repo=repo)
            result = await self.github_service.fork_repository(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["owner"] = owner
            extra["repo"] = repo
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except RepositoryError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "owner": owner, "repo": repo}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to fork repository '{owner}/{repo}': {str(e)}",
                "extra": {"error": str(e), "owner": owner, "repo": repo}
            }

    @ecp.action(name="delete_repository",
                type="GitHub", 
                description="Delete your own repository")
    async def delete_repository(self, repo: str) -> Dict[str, Any]:
        """Delete your own repository.
        
        Args:
            repo (str): The name of your repository to delete.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # Use authenticated user as owner
            owner = self.github_service.authenticated_user.login
            request = DeleteRepositoryRequest(owner=owner, repo=repo)
            result = await self.github_service.delete_repository(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["repo"] = repo
            extra["owner"] = owner
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except RepositoryError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "repo": repo}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete repository '{repo}': {str(e)}",
                "extra": {"error": str(e), "repo": repo}
            }

    # --------------- Git Operations ---------------
    @ecp.action(name="git_init", 
                type="GitHub", 
                description="Initialize a local directory as Git repository")
    async def git_init(
        self,
        local_path: str,
        remote_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initialize a local directory as Git repository.
        
        Args:
            local_path (str): The local directory path to initialize as a Git repository (relative to base_dir).
            remote_url (Optional[str]): The remote repository URL to set as origin. If None, no remote is added.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = InitRepositoryRequest(local_path=resolved_path, remote_url=remote_url)
            result = await self.github_service.init_repository(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["local_path"] = local_path
            extra["remote_url"] = remote_url
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except (GitError, RepositoryError) as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "local_path": local_path}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to initialize repository in '{local_path}': {str(e)}",
                "extra": {"error": str(e), "local_path": local_path}
            }
    
    @ecp.action(name="git_clone", 
                type="GitHub", 
                description="Clone a repository to local directory (automatically forks if not your repository)")
    async def git_clone(
        self,
        owner: str,
        repo: str,
        local_path: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Clone a repository to local directory.
        
        If the repository belongs to someone else, it will automatically fork the repository
        to your account first, then clone the forked version.
        
        Args:
            owner (str): The owner of the repository.
            repo (str): The name of the repository.
            local_path (str): The local directory path where the repository will be cloned (relative to base_dir).
            branch (Optional[str]): The specific branch to clone. If None, clones the default branch.
        
        Returns:
            Dict with success, message, and extra fields
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
                
                extra = result.extra.copy() if result.extra else {}
                extra["owner"] = owner
                extra["repo"] = repo
                extra["local_path"] = local_path
                extra["branch"] = branch
                
                return {
                    "success": result.success,
                    "message": result.message,
                    "extra": extra
                }
            
            # If it's someone else's repository, fork it first
            try:
                fork_request = ForkRepositoryRequest(owner=owner, repo=repo)
                fork_result = await self.github_service.fork_repository(fork_request)
                
                if not fork_result.success or "repository" not in fork_result.extra:
                    extra = fork_result.extra.copy() if fork_result.extra else {}
                    extra["owner"] = owner
                    extra["repo"] = repo
                    extra["local_path"] = local_path
                    return {
                        "success": False,
                        "message": f"Failed to fork repository: {fork_result.message}",
                        "extra": extra
                    }
                
                fork_msg = fork_result.message
                fork_repo = fork_result.extra["repository"]
                
                # Clone the forked repository
                clone_request = CloneRepositoryRequest(
                    owner=fork_repo.get("owner", current_user),
                    repo=fork_repo.get("name", repo),
                    local_path=resolved_path,
                    branch=branch
                )
                clone_result = await self.github_service.clone_repository(clone_request)
                
                extra = clone_result.extra.copy() if clone_result.extra else {}
                extra["owner"] = owner
                extra["repo"] = repo
                extra["local_path"] = local_path
                extra["branch"] = branch
                extra["fork_result"] = fork_result.extra
                
                return {
                    "success": clone_result.success,
                    "message": fork_msg + " " + clone_result.message,
                    "extra": extra
                }
                
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
                        
                        extra = clone_result.extra.copy() if clone_result.extra else {}
                        extra["owner"] = owner
                        extra["repo"] = repo
                        extra["local_path"] = local_path
                        extra["branch"] = branch
                        
                        return {
                            "success": clone_result.success,
                            "message": clone_result.message,
                            "extra": extra
                        }
                    except Exception:
                        pass
                raise e
                
        except (GitError, RepositoryError) as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "owner": owner, "repo": repo, "local_path": local_path}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to clone repository '{owner}/{repo}': {str(e)}",
                "extra": {"error": str(e), "owner": owner, "repo": repo, "local_path": local_path}
            }
    
    @ecp.action(name="git_commit",
                type="GitHub", 
                description="Commit changes to local repository")
    async def git_commit(
        self,
        local_path: str,
        message: str,
        add_all: bool = True
    ) -> Dict[str, Any]:
        """Commit changes to local repository.
        
        Args:
            local_path (str): The local repository path (relative to base_dir).
            message (str): The commit message.
            add_all (bool): Whether to add all changes before committing. Defaults to True.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitCommitRequest(
                local_path=resolved_path,
                message=message,
                files=None if add_all else []
            )
            result = await self.github_service.git_commit(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["local_path"] = local_path
            extra["message"] = message
            extra["add_all"] = add_all
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except GitError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "local_path": local_path}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to commit in '{local_path}': {str(e)}",
                "extra": {"error": str(e), "local_path": local_path}
            }

    @ecp.action(name="git_push",
                type="GitHub", 
                description="Push changes to remote repository")
    async def git_push(
        self,
        local_path: str,
        remote: str = "origin",
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Push changes to remote repository.
        
        Args:
            local_path (str): The local repository path (relative to base_dir).
            remote (str): The remote name. Defaults to "origin".
            branch (Optional[str]): The branch name to push. If None, pushes the current branch.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitPushRequest(
                local_path=resolved_path,
                remote=remote,
                branch=branch
            )
            result = await self.github_service.git_push(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["local_path"] = local_path
            extra["remote"] = remote
            extra["branch"] = branch
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except GitError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "local_path": local_path}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to push from '{local_path}': {str(e)}",
                "extra": {"error": str(e), "local_path": local_path}
            }

    @ecp.action(name="git_pull",
                type="GitHub", 
                description="Pull changes from remote repository")
    async def git_pull(
        self,
        local_path: str,
        remote: str = "origin",
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Pull changes from remote repository.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
            remote (str): The remote name. Defaults to "origin".
            branch (Optional[str]): The branch name to pull. If None, pulls the current branch.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitPullRequest(
                local_path=resolved_path,
                remote=remote,
                branch=branch
            )
            result = await self.github_service.git_pull(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["local_path"] = local_path
            extra["remote"] = remote
            extra["branch"] = branch
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except GitError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "local_path": local_path}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to pull to '{local_path}': {str(e)}",
                "extra": {"error": str(e), "local_path": local_path}
            }

    @ecp.action(name="git_fetch",
                type="GitHub", 
                description="Fetch changes from remote repository")
    async def git_fetch(
        self,
        local_path: str,
        remote: str = "origin"
    ) -> Dict[str, Any]:
        """Fetch changes from remote repository.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
            remote (str): The remote name. Defaults to "origin".
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitFetchRequest(
                local_path=resolved_path,
                remote=remote
            )
            result = await self.github_service.git_fetch(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["local_path"] = local_path
            extra["remote"] = remote
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except GitError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "local_path": local_path}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to fetch to '{local_path}': {str(e)}",
                "extra": {"error": str(e), "local_path": local_path}
            }

    # --------------- Branch Operations ---------------
    @ecp.action(name="git_create_branch",
                type="GitHub", 
                description="Create a new branch")
    async def git_create_branch(
        self,
        local_path: str,
        branch_name: str,
        checkout: bool = True
    ) -> Dict[str, Any]:
        """Create a new branch.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
            branch_name (str): The name of the new branch to create.
            checkout (bool): Whether to checkout the new branch after creation. Defaults to True.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitCreateBranchRequest(
                local_path=resolved_path,
                branch_name=branch_name,
                from_branch=None
            )
            result = await self.github_service.git_create_branch(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["local_path"] = local_path
            extra["branch_name"] = branch_name
            extra["checkout"] = checkout
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except GitError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "local_path": local_path, "branch_name": branch_name}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create branch '{branch_name}': {str(e)}",
                "extra": {"error": str(e), "local_path": local_path, "branch_name": branch_name}
            }

    @ecp.action(name="git_checkout_branch",
                type="GitHub", 
                description="Checkout an existing branch")
    async def git_checkout_branch(
        self,
        local_path: str,
        branch_name: str
    ) -> Dict[str, Any]:
        """Checkout an existing branch.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
            branch_name (str): The name of the branch to checkout.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitCheckoutBranchRequest(
                local_path=resolved_path,
                branch_name=branch_name
            )
            result = await self.github_service.git_checkout_branch(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["local_path"] = local_path
            extra["branch_name"] = branch_name
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except GitError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "local_path": local_path, "branch_name": branch_name}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to checkout branch '{branch_name}': {str(e)}",
                "extra": {"error": str(e), "local_path": local_path, "branch_name": branch_name}
            }

    @ecp.action(name="git_list_branches",
                type="GitHub", 
                description="List all branches")
    async def git_list_branches(self, local_path: str) -> Dict[str, Any]:
        """List all branches.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitListBranchesRequest(local_path=resolved_path)
            result = await self.github_service.git_list_branches(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["local_path"] = local_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except GitError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "local_path": local_path}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to list branches: {str(e)}",
                "extra": {"error": str(e), "local_path": local_path}
            }

    @ecp.action(name="git_delete_branch",
                type="GitHub", 
                description="Delete a branch")
    async def git_delete_branch(
        self,
        local_path: str,
        branch_name: str,
        force: bool = False
    ) -> Dict[str, Any]:
        """Delete a branch.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
            branch_name (str): The name of the branch to delete.
            force (bool): Whether to force delete the branch even if it's not merged. Defaults to False.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitDeleteBranchRequest(
                local_path=resolved_path,
                branch_name=branch_name,
                force=force
            )
            result = await self.github_service.git_delete_branch(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["local_path"] = local_path
            extra["branch_name"] = branch_name
            extra["force"] = force
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        except GitError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "local_path": local_path, "branch_name": branch_name}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete branch '{branch_name}': {str(e)}",
                "extra": {"error": str(e), "local_path": local_path, "branch_name": branch_name}
            }

    @ecp.action(name="git_status",
                type="GitHub", 
                description="Get Git repository status")
    async def git_status(self, local_path: str) -> Dict[str, Any]:
        """Get Git repository status.
        
        Args:
            local_path (str): The local repository path (relative to base_dir or absolute).
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            resolved_path = self._resolve_path(local_path)
            request = GitStatusRequest(local_path=resolved_path)
            result = await self.github_service.git_status(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["local_path"] = local_path
            
            if result.success and "status" in extra:
                status = extra["status"]
                status_text = dedent(f"""Repository Status:
                    Current Branch: {status.get('current_branch', 'Unknown')}
                    Dirty: {status.get('is_dirty', False)}
                    Modified Files: {', '.join(status.get('modified_files', [])) if status.get('modified_files') else 'None'}
                    Staged Files: {', '.join(status.get('staged_files', [])) if status.get('staged_files') else 'None'}
                    Untracked Files: {', '.join(status.get('untracked_files', [])) if status.get('untracked_files') else 'None'}
                    Branches: {', '.join(status.get('branches', []))}""")
                message = status_text
            else:
                message = result.message
            
            return {
                "success": result.success,
                "message": message,
                "extra": extra
            }
        except GitError as e:
            return {
                "success": False,
                "message": str(e),
                "extra": {"error": str(e), "local_path": local_path}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get status: {str(e)}",
                "extra": {"error": str(e), "local_path": local_path}
            }

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
                <info>
                GitHub Environment:
                Username: {self.username}
                Authenticated: {bool(self.token)}
                Service Available: {self.github_service is not None}
                Git Status: {git_status}
                </info>
            """)
            
            return {
                "state": state,
                "extra": {},
            }
        except Exception as e:
            logger.error(f"Failed to get GitHub state: {e}")
            return {
                "state": str(e),
                "extra": {
                    "error": str(e),
                },
            }