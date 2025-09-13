"""GitHub service implementation with PAT authentication."""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.environments.github.cache import GitHubCache
from src.environments.github.client import GitHubClient
from src.environments.github.types import (
    GitHubAuth,
    GitHubConfig,
    GitHubRepository,
    GitHubIssue,
    GitHubPullRequest,
    GitHubCommit,
    GitHubBranch,
    GitHubUser,
    GitHubOrganization,
    GitHubSearchResult,
    GitHubFile,
    GitHubDirectory,
    GitHubContent,
    GitHubRequest,
    GitHubResponse,
)


class GitHubService:
    """Async GitHub service with PAT authentication, caching, and comprehensive API support."""

    def __init__(self, config: GitHubConfig):
        """Initialize GitHub service.
        
        Args:
            config: GitHub service configuration
        """
        self.config = config
        self.auth = config.auth
        self.cache = GitHubCache() if config.enable_cache else None
        self._client: Optional[GitHubClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure GitHub client is initialized."""
        if self._client is None:
            self._client = GitHubClient(self.auth)
            await self._client._ensure_session()

    async def close(self) -> None:
        """Close GitHub client."""
        if self._client:
            await self._client.close()

    def _cache_key(self, operation: str, **kwargs) -> str:
        """Generate cache key for operation."""
        key_parts = [operation]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}:{v}")
        return "|".join(key_parts)

    async def _get_cached_or_fetch(self, cache_key: str, cache_type: str, fetch_func, *args, **kwargs):
        """Get from cache or fetch and cache result."""
        if not self.cache:
            return await fetch_func(*args, **kwargs)
        
        cached = await self.cache.get(cache_key, cache_type)
        if cached is not None:
            return cached
        
        result = await fetch_func(*args, **kwargs)
        await self.cache.put(cache_key, result, cache_type)
        return result

    # --------------- Authentication & User Info ---------------
    async def verify_authentication(self) -> GitHubUser:
        """Verify PAT authentication and get user info."""
        await self._ensure_client()
        
        cache_key = self._cache_key("auth", token=self.auth.token[:10])
        return await self._get_cached_or_fetch(
            cache_key, 'user', self._fetch_user_info
        )

    async def _fetch_user_info(self) -> GitHubUser:
        """Fetch authenticated user information."""
        response = await self._client.get('/user')
        return GitHubUser(**response.data)

    async def get_user(self, username: str) -> GitHubUser:
        """Get user information by username."""
        await self._ensure_client()
        
        cache_key = self._cache_key("user", username=username)
        return await self._get_cached_or_fetch(
            cache_key, 'user', self._fetch_user, username
        )

    async def _fetch_user(self, username: str) -> GitHubUser:
        """Fetch user information."""
        response = await self._client.get(f'/users/{username}')
        return GitHubUser(**response.data)

    async def get_organization(self, org_name: str) -> GitHubOrganization:
        """Get organization information."""
        await self._ensure_client()
        
        cache_key = self._cache_key("org", org=org_name)
        return await self._get_cached_or_fetch(
            cache_key, 'user', self._fetch_organization, org_name
        )

    async def _fetch_organization(self, org_name: str) -> GitHubOrganization:
        """Fetch organization information."""
        response = await self._client.get(f'/orgs/{org_name}')
        return GitHubOrganization(**response.data)

    # --------------- Repository Operations ---------------
    async def get_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Get repository information."""
        await self._ensure_client()
        
        cache_key = self._cache_key("repo", owner=owner, repo=repo)
        return await self._get_cached_or_fetch(
            cache_key, 'repo', self._fetch_repository, owner, repo
        )

    async def _fetch_repository(self, owner: str, repo: str) -> GitHubRepository:
        """Fetch repository information."""
        response = await self._client.get(f'/repos/{owner}/{repo}')
        return GitHubRepository(**response.data)

    async def list_repositories(
        self,
        owner: Optional[str] = None,
        repo_type: str = "all",
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 30
    ) -> List[GitHubRepository]:
        """List repositories for user or organization."""
        await self._ensure_client()
        
        if owner:
            endpoint = f'/users/{owner}/repos'
            cache_key = self._cache_key("repos", owner=owner, type=repo_type, sort=sort)
        else:
            endpoint = '/user/repos'
            cache_key = self._cache_key("user_repos", type=repo_type, sort=sort)
        
        params = {
            'type': repo_type,
            'sort': sort,
            'direction': direction,
            'per_page': per_page
        }
        
        return await self._get_cached_or_fetch(
            cache_key, 'repo', self._fetch_repositories, endpoint, params
        )

    async def _fetch_repositories(self, endpoint: str, params: Dict[str, Any]) -> List[GitHubRepository]:
        """Fetch repositories."""
        response = await self._client.get(endpoint, params=params)
        return [GitHubRepository(**repo) for repo in response.data]

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
        await self._ensure_client()
        
        data = {
            'name': name,
            'description': description,
            'private': private,
            'auto_init': auto_init,
            'gitignore_template': gitignore_template,
            'license_template': license_template
        }
        
        response = await self._client.post('/user/repos', data=data)
        repo = GitHubRepository(**response.data)
        
        # Invalidate cache
        if self.cache:
            await self.cache.clear('repo')
        
        return repo

    # --------------- File Operations ---------------
    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None
    ) -> GitHubFile:
        """Get file content from repository."""
        await self._ensure_client()
        
        cache_key = self._cache_key("file", owner=owner, repo=repo, path=path, ref=ref)
        return await self._get_cached_or_fetch(
            cache_key, 'content', self._fetch_file_content, owner, repo, path, ref
        )

    async def _fetch_file_content(self, owner: str, repo: str, path: str, ref: Optional[str]) -> GitHubFile:
        """Fetch file content."""
        params = {'ref': ref} if ref else {}
        response = await self._client.get(f'/repos/{owner}/{repo}/contents/{path}', params=params)
        return GitHubFile(**response.data)

    async def get_directory_contents(
        self,
        owner: str,
        repo: str,
        path: str = "",
        ref: Optional[str] = None
    ) -> List[Union[GitHubFile, GitHubDirectory]]:
        """Get directory contents."""
        await self._ensure_client()
        
        cache_key = self._cache_key("dir", owner=owner, repo=repo, path=path, ref=ref)
        return await self._get_cached_or_fetch(
            cache_key, 'content', self._fetch_directory_contents, owner, repo, path, ref
        )

    async def _fetch_directory_contents(
        self, owner: str, repo: str, path: str, ref: Optional[str]
    ) -> List[Union[GitHubFile, GitHubDirectory]]:
        """Fetch directory contents."""
        params = {'ref': ref} if ref else {}
        response = await self._client.get(f'/repos/{owner}/{repo}/contents/{path}', params=params)
        
        contents = []
        for item in response.data:
            if item['type'] == 'file':
                contents.append(GitHubFile(**item))
            elif item['type'] == 'dir':
                contents.append(GitHubDirectory(**item))
        
        return contents

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
        await self._ensure_client()
        
        # Encode content to base64
        content_bytes = content.encode('utf-8')
        content_b64 = base64.b64encode(content_bytes).decode('ascii')
        
        data = {
            'message': message,
            'content': content_b64,
            'branch': branch
        }
        
        if sha:
            data['sha'] = sha
        
        response = await self._client.put(f'/repos/{owner}/{repo}/contents/{path}', data=data)
        
        # Invalidate cache
        if self.cache:
            await self.cache.delete(
                self._cache_key("file", owner=owner, repo=repo, path=path),
                'content'
            )
            await self.cache.delete(
                self._cache_key("dir", owner=owner, repo=repo, path=str(Path(path).parent)),
                'content'
            )
        
        return response.data

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
        await self._ensure_client()
        
        data = {
            'message': message,
            'sha': sha,
            'branch': branch
        }
        
        response = await self._client.delete(f'/repos/{owner}/{repo}/contents/{path}', data=data)
        
        # Invalidate cache
        if self.cache:
            await self.cache.delete(
                self._cache_key("file", owner=owner, repo=repo, path=path),
                'content'
            )
            await self.cache.delete(
                self._cache_key("dir", owner=owner, repo=repo, path=str(Path(path).parent)),
                'content'
            )
        
        return response.data

    # --------------- Branch Operations ---------------
    async def list_branches(self, owner: str, repo: str) -> List[GitHubBranch]:
        """List repository branches."""
        await self._ensure_client()
        
        cache_key = self._cache_key("branches", owner=owner, repo=repo)
        return await self._get_cached_or_fetch(
            cache_key, 'repo', self._fetch_branches, owner, repo
        )

    async def _fetch_branches(self, owner: str, repo: str) -> List[GitHubBranch]:
        """Fetch repository branches."""
        response = await self._client.get(f'/repos/{owner}/{repo}/branches')
        return [GitHubBranch(**branch) for branch in response.data]

    async def get_branch(self, owner: str, repo: str, branch: str) -> GitHubBranch:
        """Get specific branch information."""
        await self._ensure_client()
        
        cache_key = self._cache_key("branch", owner=owner, repo=repo, branch=branch)
        return await self._get_cached_or_fetch(
            cache_key, 'repo', self._fetch_branch, owner, repo, branch
        )

    async def _fetch_branch(self, owner: str, repo: str, branch: str) -> GitHubBranch:
        """Fetch specific branch."""
        response = await self._client.get(f'/repos/{owner}/{repo}/branches/{branch}')
        return GitHubBranch(**response.data)

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
        await self._ensure_client()
        
        params = {
            'state': state,
            'per_page': per_page
        }
        
        if labels:
            params['labels'] = ','.join(labels)
        if assignee:
            params['assignee'] = assignee
        if creator:
            params['creator'] = creator
        
        response = await self._client.get(f'/repos/{owner}/{repo}/issues', params=params)
        return [GitHubIssue(**issue) for issue in response.data]

    async def get_issue(self, owner: str, repo: str, issue_number: int) -> GitHubIssue:
        """Get specific issue."""
        await self._ensure_client()
        
        cache_key = self._cache_key("issue", owner=owner, repo=repo, number=issue_number)
        return await self._get_cached_or_fetch(
            cache_key, 'default', self._fetch_issue, owner, repo, issue_number
        )

    async def _fetch_issue(self, owner: str, repo: str, issue_number: int) -> GitHubIssue:
        """Fetch specific issue."""
        response = await self._client.get(f'/repos/{owner}/{repo}/issues/{issue_number}')
        return GitHubIssue(**response.data)

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
        await self._ensure_client()
        
        data = {
            'title': title,
            'body': body,
            'assignees': assignees or [],
            'labels': labels or []
        }
        
        response = await self._client.post(f'/repos/{owner}/{repo}/issues', data=data)
        return GitHubIssue(**response.data)

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
        await self._ensure_client()
        
        params = {
            'state': state,
            'per_page': per_page
        }
        
        if head:
            params['head'] = head
        if base:
            params['base'] = base
        
        response = await self._client.get(f'/repos/{owner}/{repo}/pulls', params=params)
        return [GitHubPullRequest(**pr) for pr in response.data]

    async def get_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest:
        """Get specific pull request."""
        await self._ensure_client()
        
        cache_key = self._cache_key("pr", owner=owner, repo=repo, number=pr_number)
        return await self._get_cached_or_fetch(
            cache_key, 'default', self._fetch_pull_request, owner, repo, pr_number
        )

    async def _fetch_pull_request(self, owner: str, repo: str, pr_number: int) -> GitHubPullRequest:
        """Fetch specific pull request."""
        response = await self._client.get(f'/repos/{owner}/{repo}/pulls/{pr_number}')
        return GitHubPullRequest(**response.data)

    # --------------- Search Operations ---------------
    async def search_repositories(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 30
    ) -> GitHubSearchResult:
        """Search repositories."""
        await self._ensure_client()
        
        cache_key = self._cache_key("search_repos", query=query, sort=sort, order=order)
        return await self._get_cached_or_fetch(
            cache_key, 'search', self._search_repositories, query, sort, order, per_page
        )

    async def _search_repositories(self, query: str, sort: str, order: str, per_page: int) -> GitHubSearchResult:
        """Search repositories."""
        params = {
            'q': query,
            'sort': sort,
            'order': order,
            'per_page': per_page
        }
        
        response = await self._client.get('/search/repositories', params=params)
        return GitHubSearchResult(**response.data)

    async def search_issues(
        self,
        query: str,
        sort: str = "updated",
        order: str = "desc",
        per_page: int = 30
    ) -> GitHubSearchResult:
        """Search issues."""
        await self._ensure_client()
        
        cache_key = self._cache_key("search_issues", query=query, sort=sort, order=order)
        return await self._get_cached_or_fetch(
            cache_key, 'search', self._search_issues, query, sort, order, per_page
        )

    async def _search_issues(self, query: str, sort: str, order: str, per_page: int) -> GitHubSearchResult:
        """Search issues."""
        params = {
            'q': query,
            'sort': sort,
            'order': order,
            'per_page': per_page
        }
        
        response = await self._client.get('/search/issues', params=params)
        return GitHubSearchResult(**response.data)

    # --------------- Utility Methods ---------------
    async def get_rate_limit_info(self) -> Dict[str, Any]:
        """Get current rate limit information."""
        await self._ensure_client()
        
        response = await self._client.get('/rate_limit')
        return response.data

    async def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """Clear service cache."""
        if self.cache:
            await self.cache.clear(cache_type)

    async def cleanup_expired_cache(self) -> int:
        """Cleanup expired cache entries."""
        if self.cache:
            return await self.cache.cleanup_expired()
        return 0
