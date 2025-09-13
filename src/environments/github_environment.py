"""GitHub environment implementation for AgentWorld."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from src.environments.github import (
    GitHubService,
    GitHubAuth,
    GitHubConfig,
    GitHubRepository,
    GitHubIssue,
    GitHubPullRequest,
    GitHubCommit,
    GitHubBranch,
    GitHubUser,
    GitHubFile,
    GitHubDirectory,
    GitHubSearchResult,
    GitHubError,
    AuthenticationError,
    NotFoundError,
)
from src.logger import logger
from src.environments.protocol.environment import BaseEnvironment
from src.environments.protocol import ecp


_GITHUB_ENVIRONMENT_RULES = """<environment_github>
<state>
The environment state includes:
1. Repository: GitHub repository information
2. Branch: GitHub branch information
3. Issue: GitHub issue information
4. Pull Request: GitHub pull request information
</state>

<vision>
No vision available.
</vision>

<interaction>
Available actions:
</interaction>
</environment_github>
"""

@ecp.environment(name = "github",
                 env_type = "github",
                 description = "GitHub environment for interacting with GitHub repositories and services",
                 rules = _GITHUB_ENVIRONMENT_RULES)
class GitHubEnvironment(BaseEnvironment):
    """GitHub environment for interacting with GitHub repositories and services."""

    def __init__(
        self,
        token: str ,
        username: Optional[str] = None,
        base_url: str = "https://api.github.com",
        timeout: int = 30,
        enable_cache: bool = True,
        cache_ttl: int = 300,
        **kwargs
    ):
        """Initialize GitHub environment.
        
        Args:
            token: GitHub Personal Access Token (PAT)
            username: GitHub username (optional, will be fetched if not provided)
            base_url: GitHub API base URL
            timeout: Request timeout in seconds
            enable_cache: Enable caching for API responses
            cache_ttl: Cache time-to-live in seconds
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        # Initialize GitHub authentication and configuration
        self.auth = GitHubAuth(
            token=token,
            username=username,
            base_url=base_url,
            timeout=timeout
        )
        
        self.config = GitHubConfig(
            auth=self.auth,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl,
            **kwargs
        )
        
        self._service: Optional[GitHubService] = None
        self._authenticated_user: Optional[GitHubUser] = None

    async def initialize(self) -> None:
        """Initialize the GitHub environment."""
        try:
            self._service = GitHubService(self.config)
            await self._service.__aenter__()
            
            # Verify authentication and get user info
            self._authenticated_user = await self._service.verify_authentication()
            logger.debug(f"| GitHub environment initialized for user: {self._authenticated_user.login}")
            
        except AuthenticationError as e:
            logger.error(f"| GitHub authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"| Failed to initialize GitHub environment: {e}")
            raise

    async def cleanup(self) -> None:
        """Cleanup the GitHub environment."""
        if self._service:
            await self._service.__aexit__(None, None, None)
            self._service = None
        logger.info("GitHub environment cleaned up")

    @property
    def service(self) -> GitHubService:
        """Get GitHub service instance."""
        if self._service is None:
            raise RuntimeError("GitHub environment not initialized")
        return self._service

    @property
    def authenticated_user(self) -> GitHubUser:
        """Get authenticated user information."""
        if self._authenticated_user is None:
            raise RuntimeError("GitHub environment not initialized")
        return self._authenticated_user

    # --------------- Repository Operations ---------------
    @ecp.action(name = "get_repository",
                description = "Get repository information.")
    async def _get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Get repository information.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name

        Returns:
            GitHubRepository: GitHub repository information
        """
        try:
            return await self.service.get_repository(owner, repo)
        except NotFoundError:
            logger.error(f"| Repository not found: {owner}/{repo}")
            raise
        except Exception as e:
            logger.error(f"| Failed to get repository {owner}/{repo}: {e}")
            raise

    @ecp.action(name = "list_repositories",
                description = "List repositories.")
    async def _list_repositories(
        self,
        owner: Optional[str] = None,
        repo_type: str = "all",
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 30
    ) -> List[GitHubRepository]:
        """List repositories.
        
        Args:
            owner (str): GitHub repository owner
            repo_type (str): GitHub repository type
            sort (str): GitHub repository sort
            direction (str): GitHub repository direction
            per_page (int): GitHub repository per page

        Returns:
            List[GitHubRepository]: List of GitHub repositories
        """
        try:
            return await self.service.list_repositories(
                owner=owner,
                repo_type=repo_type,
                sort=sort,
                direction=direction,
                per_page=per_page
            )
        except Exception as e:
            logger.error(f"| Failed to list repositories: {e}")
            raise
    
    @ecp.action(name = "create_repository",
                description = "Create a new repository.")
    async def _create_repository(
        self,
        name: str,
        description: Optional[str] = None,
        private: bool = False,
        auto_init: bool = False,
        gitignore_template: Optional[str] = None,
        license_template: Optional[str] = None
    ) -> GitHubRepository:
        """Create a new repository.
        
        Args:
            name (str): GitHub repository name
            description (str): GitHub repository description
            private (bool): GitHub repository private
            auto_init (bool): GitHub repository auto init
            gitignore_template (str): GitHub repository gitignore template
            license_template (str): GitHub repository license template

        Returns:
            GitHubRepository: GitHub repository information
        """
        try:
            repo = await self.service.create_repository(
                name=name,
                description=description,
                private=private,
                auto_init=auto_init,
                gitignore_template=gitignore_template,
                license_template=license_template
            )
            logger.info(f"| Created repository: {repo.full_name}")
            return repo
        except Exception as e:
            logger.error(f"| Failed to create repository {name}: {e}")
            raise

    # --------------- File Operations ---------------
    @ecp.action(name = "get_file_content",
                description = "Get file content from repository.")
    async def _get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None
    ) -> GitHubFile:
        """Get file content from repository.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            path (str): GitHub repository path
            ref (str): GitHub repository reference

        Returns:
            GitHubFile: GitHub file information
        """
        try:
            return await self.service.get_file_content(owner, repo, path, ref)
        except NotFoundError:
            logger.error(f"| File not found: {owner}/{repo}/{path}")
            raise
        except Exception as e:
            logger.error(f"| Failed to get file content {owner}/{repo}/{path}: {e}")
            raise

    @ecp.action(name = "get_directory_contents",
                description = "Get directory contents from repository.")
    async def _get_directory_contents(
        self,
        owner: str,
        repo: str,
        path: str = "",
        ref: Optional[str] = None
    ) -> List[Union[GitHubFile, GitHubDirectory]]:
        """Get directory contents.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            path (str): GitHub repository path
            ref (str): GitHub repository reference

        Returns:
            List[Union[GitHubFile, GitHubDirectory]]: List of GitHub file and directory information
        """
        try:
            return await self.service.get_directory_contents(owner, repo, path, ref)
        except NotFoundError:
            logger.error(f"| Directory not found: {owner}/{repo}/{path}")
            raise
        except Exception as e:
            logger.error(f"| Failed to get directory contents {owner}/{repo}/{path}: {e}")
            raise

    @ecp.action(name = "create_or_update_file",
                description = "Create or update a file in repository.")
    async def _create_or_update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: Optional[str] = None,
        sha: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create or update a file in repository.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            path (str): GitHub repository path
            content (str): GitHub repository content
            message (str): GitHub repository message
            branch (str): GitHub repository branch
            sha (str): GitHub repository sha
        
        Returns:
            Dict[str, Any]: GitHub repository information
        """
        try:
            result = await self.service.create_or_update_file(
                owner, repo, path, content, message, branch, sha
            )
            logger.info(f"| File {'updated' if sha else 'created'}: {owner}/{repo}/{path}")
            return result
        except Exception as e:
            logger.error(f"| Failed to create/update file {owner}/{repo}/{path}: {e}")
            raise

    @ecp.action(name = "delete_file",
                description = "Delete a file from repository.")
    async def _delete_file(
        self,
        owner: str,
        repo: str,
        path: str,
        message: str,
        sha: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete a file from repository.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            path (str): GitHub repository path
            message (str): GitHub repository message
            sha (str): GitHub repository sha
            branch (str): GitHub repository branch
        
        Returns:
            Dict[str, Any]: GitHub repository information
        """
        try:
            result = await self.service.delete_file(owner, repo, path, message, sha, branch)
            logger.info(f"| File deleted: {owner}/{repo}/{path}")
            return result
        except Exception as e:
            logger.error(f"| Failed to delete file {owner}/{repo}/{path}: {e}")
            raise

    # --------------- Branch Operations ---------------
    @ecp.action(name = "list_branches",
                description = "List repository branches.")
    async def _list_branches(self, owner: str, repo: str) -> List[GitHubBranch]:
        """List repository branches.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name

        Returns:
            List[GitHubBranch]: List of GitHub branches
        """
        try:
            return await self.service.list_branches(owner, repo)
        except Exception as e:
            logger.error(f"| Failed to list branches for {owner}/{repo}: {e}")
            raise

    @ecp.action(name = "get_branch",
                description = "Get specific branch information.")
    async def _get_branch(self, owner: str, repo: str, branch: str) -> GitHubBranch:
        """Get specific branch information.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            branch (str): GitHub repository branch
        
        Returns:
            GitHubBranch: GitHub branch information
        """
        try:
            return await self.service.get_branch(owner, repo, branch)
        except NotFoundError:
            logger.error(f"| Branch not found: {owner}/{repo}/{branch}")
            raise
        except Exception as e:
            logger.error(f"| Failed to get branch {owner}/{repo}/{branch}: {e}")
            raise

    # --------------- Issue Operations ---------------
    @ecp.action(name = "list_issues",
                description = "List repository issues.")
    async def _list_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        creator: Optional[str] = None,
        per_page: int = 30
    ) -> List[GitHubIssue]:
        """List repository issues.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            state (str): GitHub repository state
            labels (List[str]): GitHub repository labels
            assignee (str): GitHub repository assignee
            creator (str): GitHub repository creator
            per_page (int): GitHub repository per page

        Returns:
            List[GitHubIssue]: List of GitHub issues
        """
        try:
            return await self.service.list_issues(
                owner, repo, state, labels, assignee, creator, per_page
            )
        except Exception as e:
            logger.error(f"| Failed to list issues for {owner}/{repo}: {e}")
            raise

    @ecp.action(name = "get_issue",
                description = "Get specific issue.")
    async def _get_issue(self, owner: str, repo: str, issue_number: int) -> GitHubIssue:
        """Get specific issue.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            issue_number (int): GitHub repository issue number

        Returns:
            GitHubIssue: GitHub issue information
        """
        try:
            return await self.service.get_issue(owner, repo, issue_number)
        except NotFoundError:
            logger.error(f"| Issue not found: {owner}/{repo}#{issue_number}")
            raise
        except Exception as e:
            logger.error(f"| Failed to get issue {owner}/{repo}#{issue_number}: {e}")
            raise

    @ecp.action(name = "create_issue",
                description = "Create a new issue.")
    async def _create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: Optional[str] = None,
        assignees: Optional[List[str]] = None,
        labels: Optional[List[str]] = None
    ) -> GitHubIssue:
        """Create a new issue.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            title (str): GitHub repository title
            body (str): GitHub repository body
            assignees (List[str]): GitHub repository assignees
            labels (List[str]): GitHub repository labels

        Returns:
            GitHubIssue: GitHub issue information
        """
        try:
            issue = await self.service.create_issue(
                owner, repo, title, body, assignees, labels
            )
            logger.info(f"| Created issue: {owner}/{repo}#{issue.number}")
            return issue
        except Exception as e:
            logger.error(f"| Failed to create issue in {owner}/{repo}: {e}")
            raise

    # --------------- Pull Request Operations ---------------
    @ecp.action(name = "list_pull_requests",
                description = "List repository pull requests.")
    async def _list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        head: Optional[str] = None,
        base: Optional[str] = None,
        per_page: int = 30
    ) -> List[GitHubPullRequest]:
        """List repository pull requests.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            state (str): GitHub repository state
            head (str): GitHub repository head
            base (str): GitHub repository base
            per_page (int): GitHub repository per page

        Returns:
            List[GitHubPullRequest]: List of GitHub pull requests
        """
        try:
            return await self.service.list_pull_requests(
                owner, repo, state, head, base, per_page
            )
        except Exception as e:
            logger.error(f"| Failed to list pull requests for {owner}/{repo}: {e}")
            raise

    @ecp.action(name = "get_pull_request",
                description = "Get specific pull request.")
    async def _get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest:
        """Get specific pull request.
        
        Args:
            owner (str): GitHub repository owner
            repo (str): GitHub repository name
            pr_number (int): GitHub repository pull request number

        Returns:
            GitHubPullRequest: GitHub pull request information
        """
        try:
            return await self.service.get_pull_request(owner, repo, pr_number)
        except NotFoundError:
            logger.error(f"| Pull request not found: {owner}/{repo}#{pr_number}")
            raise
        except Exception as e:
            logger.error(f"| Failed to get pull request {owner}/{repo}#{pr_number}: {e}")
            raise

    # --------------- Search Operations ---------------
    @ecp.action(name = "search_repositories",
                description = "Search repositories.")
    async def _search_repositories(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 30
    ) -> GitHubSearchResult:
        """Search repositories.
        
        Args:
            query (str): GitHub repository query
            sort (str): GitHub repository sort
            order (str): GitHub repository order
            per_page (int): GitHub repository per page
        
        Returns:
            GitHubSearchResult: GitHub search result information
        """
        try:
            return await self.service.search_repositories(query, sort, order, per_page)
        except Exception as e:
            logger.error(f"| Failed to search repositories with query '{query}': {e}")
            raise

    @ecp.action(name = "search_issues",
                description = "Search issues.")
    async def _search_issues(
        self,
        query: str,
        sort: str = "updated",
        order: str = "desc",
        per_page: int = 30
    ) -> GitHubSearchResult:
        """Search issues.
        
        Args:
            query (str): GitHub repository query
            sort (str): GitHub repository sort
            order (str): GitHub repository order
            per_page (int): GitHub repository per page

        Returns:
            GitHubSearchResult: GitHub search result information
        """
        try:
            return await self.service.search_issues(query, sort, order, per_page)
        except Exception as e:
            logger.error(f"| Failed to search issues with query '{query}': {e}")
            raise

    # --------------- User Operations ---------------
    @ecp.action(name = "get_user",
                description = "Get user information by username.")
    async def _get_user(self, username: str) -> GitHubUser:
        """Get user information by username.
        
        Args:
            username (str): GitHub repository username

        Returns:
            GitHubUser: GitHub user information
        """
        try:
            return await self.service.get_user(username)
        except NotFoundError:
            logger.error(f"| User not found: {username}")
            raise
        except Exception as e:
            logger.error(f"| Failed to get user {username}: {e}")
            raise

    # --------------- Utility Methods ---------------
    @ecp.action(name = "get_rate_limit_info",
                description = "Get current rate limit information.")
    async def _get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information.
        
        Returns:
            Dict[str, Any]: GitHub rate limit information
        """
        try:
            return await self.service.get_rate_limit_info()
        except Exception as e:
            logger.error(f"| Failed to get rate limit info: {e}")
            raise

    @ecp.action(name = "clear_cache",
                description = "Clear service cache.")
    async def _clear_cache(self, cache_type: Optional[str] = None) -> None:
        """Clear service cache.
        
        Args:
            cache_type (str): GitHub repository cache type
        """
        try:
            await self.service.clear_cache(cache_type)
            logger.info(f"| Cache cleared: {cache_type or 'all'}")
        except Exception as e:
            logger.error(f"| Failed to clear cache: {e}")
            raise

    @ecp.action(name = "cleanup_expired_cache",
                description = "Cleanup expired cache entries.")
    async def _cleanup_expired_cache(self) -> int:
        """Cleanup expired cache entries.
        
        Returns:
            int: GitHub repository expired cache entries
        """
        try:
            count = await self.service.cleanup_expired_cache()
            if count > 0:
                logger.info(f"| Cleaned up {count} expired cache entries")
            return count
        except Exception as e:
            logger.error(f"| Failed to cleanup expired cache: {e}")
            raise

    # --------------- Environment Interface Methods ---------------
    async def get_info(self) -> Dict[str, Any]:
        """Get environment information.
        
        Returns:
            Dict[str, Any]: GitHub environment information
        """
        return {
            "type": "github",
            "authenticated_user": self.authenticated_user.login if self._authenticated_user else None,
            "base_url": self.auth.base_url,
            "cache_enabled": self.config.enable_cache,
            "cache_ttl": self.config.cache_ttl,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Dict[str, Any]: GitHub health check information
        """
        try:
            rate_limit = await self.get_rate_limit_info()
            return {
                "status": "healthy",
                "authenticated": self._authenticated_user is not None,
                "rate_limit_remaining": rate_limit.get("rate", {}).get("remaining"),
                "rate_limit_reset": rate_limit.get("rate", {}).get("reset"),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
            
    async def get_state(self) -> Dict[str, Any]:
        """Get environment state.
        
        Returns:
            Dict[str, Any]: GitHub environment state
        """
        state: Dict[str, Any] = {
            "state": "GitHub environment is ready to use."
        }
        return state
