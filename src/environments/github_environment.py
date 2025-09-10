"""GitHub environment implementation for AgentWorld."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.environments.environment.github import (
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
from src.logger.log import get_logger

logger = get_logger(__name__)


class GitHubEnvironment():
    """GitHub environment for interacting with GitHub repositories and services."""

    def __init__(
        self,
        token: str,
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
            logger.info(f"GitHub environment initialized for user: {self._authenticated_user.login}")
            
        except AuthenticationError as e:
            logger.error(f"GitHub authentication failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize GitHub environment: {e}")
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
    async def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Get repository information."""
        try:
            return await self.service.get_repository(owner, repo)
        except NotFoundError:
            logger.error(f"Repository not found: {owner}/{repo}")
            raise
        except Exception as e:
            logger.error(f"Failed to get repository {owner}/{repo}: {e}")
            raise

    async def list_repositories(
        self,
        owner: Optional[str] = None,
        repo_type: str = "all",
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 30
    ) -> List[GitHubRepository]:
        """List repositories."""
        try:
            return await self.service.list_repositories(
                owner=owner,
                repo_type=repo_type,
                sort=sort,
                direction=direction,
                per_page=per_page
            )
        except Exception as e:
            logger.error(f"Failed to list repositories: {e}")
            raise

    async def create_repository(
        self,
        name: str,
        description: Optional[str] = None,
        private: bool = False,
        auto_init: bool = False,
        gitignore_template: Optional[str] = None,
        license_template: Optional[str] = None
    ) -> GitHubRepository:
        """Create a new repository."""
        try:
            repo = await self.service.create_repository(
                name=name,
                description=description,
                private=private,
                auto_init=auto_init,
                gitignore_template=gitignore_template,
                license_template=license_template
            )
            logger.info(f"Created repository: {repo.full_name}")
            return repo
        except Exception as e:
            logger.error(f"Failed to create repository {name}: {e}")
            raise

    # --------------- File Operations ---------------
    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None
    ) -> GitHubFile:
        """Get file content from repository."""
        try:
            return await self.service.get_file_content(owner, repo, path, ref)
        except NotFoundError:
            logger.error(f"File not found: {owner}/{repo}/{path}")
            raise
        except Exception as e:
            logger.error(f"Failed to get file content {owner}/{repo}/{path}: {e}")
            raise

    async def get_directory_contents(
        self,
        owner: str,
        repo: str,
        path: str = "",
        ref: Optional[str] = None
    ) -> List[Union[GitHubFile, GitHubDirectory]]:
        """Get directory contents."""
        try:
            return await self.service.get_directory_contents(owner, repo, path, ref)
        except NotFoundError:
            logger.error(f"Directory not found: {owner}/{repo}/{path}")
            raise
        except Exception as e:
            logger.error(f"Failed to get directory contents {owner}/{repo}/{path}: {e}")
            raise

    async def create_or_update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: Optional[str] = None,
        sha: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create or update a file in repository."""
        try:
            result = await self.service.create_or_update_file(
                owner, repo, path, content, message, branch, sha
            )
            logger.info(f"File {'updated' if sha else 'created'}: {owner}/{repo}/{path}")
            return result
        except Exception as e:
            logger.error(f"Failed to create/update file {owner}/{repo}/{path}: {e}")
            raise

    async def delete_file(
        self,
        owner: str,
        repo: str,
        path: str,
        message: str,
        sha: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Delete a file from repository."""
        try:
            result = await self.service.delete_file(owner, repo, path, message, sha, branch)
            logger.info(f"File deleted: {owner}/{repo}/{path}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete file {owner}/{repo}/{path}: {e}")
            raise

    # --------------- Branch Operations ---------------
    async def list_branches(self, owner: str, repo: str) -> List[GitHubBranch]:
        """List repository branches."""
        try:
            return await self.service.list_branches(owner, repo)
        except Exception as e:
            logger.error(f"Failed to list branches for {owner}/{repo}: {e}")
            raise

    async def get_branch(self, owner: str, repo: str, branch: str) -> GitHubBranch:
        """Get specific branch information."""
        try:
            return await self.service.get_branch(owner, repo, branch)
        except NotFoundError:
            logger.error(f"Branch not found: {owner}/{repo}/{branch}")
            raise
        except Exception as e:
            logger.error(f"Failed to get branch {owner}/{repo}/{branch}: {e}")
            raise

    # --------------- Issue Operations ---------------
    async def list_issues(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        labels: Optional[List[str]] = None,
        assignee: Optional[str] = None,
        creator: Optional[str] = None,
        per_page: int = 30
    ) -> List[GitHubIssue]:
        """List repository issues."""
        try:
            return await self.service.list_issues(
                owner, repo, state, labels, assignee, creator, per_page
            )
        except Exception as e:
            logger.error(f"Failed to list issues for {owner}/{repo}: {e}")
            raise

    async def get_issue(self, owner: str, repo: str, issue_number: int) -> GitHubIssue:
        """Get specific issue."""
        try:
            return await self.service.get_issue(owner, repo, issue_number)
        except NotFoundError:
            logger.error(f"Issue not found: {owner}/{repo}#{issue_number}")
            raise
        except Exception as e:
            logger.error(f"Failed to get issue {owner}/{repo}#{issue_number}: {e}")
            raise

    async def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: Optional[str] = None,
        assignees: Optional[List[str]] = None,
        labels: Optional[List[str]] = None
    ) -> GitHubIssue:
        """Create a new issue."""
        try:
            issue = await self.service.create_issue(
                owner, repo, title, body, assignees, labels
            )
            logger.info(f"Created issue: {owner}/{repo}#{issue.number}")
            return issue
        except Exception as e:
            logger.error(f"Failed to create issue in {owner}/{repo}: {e}")
            raise

    # --------------- Pull Request Operations ---------------
    async def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "open",
        head: Optional[str] = None,
        base: Optional[str] = None,
        per_page: int = 30
    ) -> List[GitHubPullRequest]:
        """List repository pull requests."""
        try:
            return await self.service.list_pull_requests(
                owner, repo, state, head, base, per_page
            )
        except Exception as e:
            logger.error(f"Failed to list pull requests for {owner}/{repo}: {e}")
            raise

    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest:
        """Get specific pull request."""
        try:
            return await self.service.get_pull_request(owner, repo, pr_number)
        except NotFoundError:
            logger.error(f"Pull request not found: {owner}/{repo}#{pr_number}")
            raise
        except Exception as e:
            logger.error(f"Failed to get pull request {owner}/{repo}#{pr_number}: {e}")
            raise

    # --------------- Search Operations ---------------
    async def search_repositories(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 30
    ) -> GitHubSearchResult:
        """Search repositories."""
        try:
            return await self.service.search_repositories(query, sort, order, per_page)
        except Exception as e:
            logger.error(f"Failed to search repositories with query '{query}': {e}")
            raise

    async def search_issues(
        self,
        query: str,
        sort: str = "updated",
        order: str = "desc",
        per_page: int = 30
    ) -> GitHubSearchResult:
        """Search issues."""
        try:
            return await self.service.search_issues(query, sort, order, per_page)
        except Exception as e:
            logger.error(f"Failed to search issues with query '{query}': {e}")
            raise

    # --------------- User Operations ---------------
    async def get_user(self, username: str) -> GitHubUser:
        """Get user information by username."""
        try:
            return await self.service.get_user(username)
        except NotFoundError:
            logger.error(f"User not found: {username}")
            raise
        except Exception as e:
            logger.error(f"Failed to get user {username}: {e}")
            raise

    # --------------- Utility Methods ---------------
    async def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information."""
        try:
            return await self.service.get_rate_limit_info()
        except Exception as e:
            logger.error(f"Failed to get rate limit info: {e}")
            raise

    async def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """Clear service cache."""
        try:
            await self.service.clear_cache(cache_type)
            logger.info(f"Cache cleared: {cache_type or 'all'}")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise

    async def cleanup_expired_cache(self) -> int:
        """Cleanup expired cache entries."""
        try:
            count = await self.service.cleanup_expired_cache()
            if count > 0:
                logger.info(f"Cleaned up {count} expired cache entries")
            return count
        except Exception as e:
            logger.error(f"Failed to cleanup expired cache: {e}")
            raise

    # --------------- Environment Interface Methods ---------------
    async def get_info(self) -> Dict[str, Any]:
        """Get environment information."""
        return {
            "type": "github",
            "authenticated_user": self.authenticated_user.login if self._authenticated_user else None,
            "base_url": self.auth.base_url,
            "cache_enabled": self.config.enable_cache,
            "cache_ttl": self.config.cache_ttl,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
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
