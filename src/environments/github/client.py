"""GitHub API client implementation."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

import aiohttp
from aiohttp import ClientSession, ClientTimeout

from src.environments.github.exceptions import (
    GitHubError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ConflictError,
    ValidationError,
    APIError,
    NetworkError,
    TimeoutError,
    PermissionError,
)
from .types import GitHubResponse, GitHubAuth


class GitHubClient:
    """Async GitHub API client with authentication and rate limiting."""
    
    def __init__(self, auth: GitHubAuth):
        """Initialize GitHub client.
        
        Args:
            auth: GitHub authentication configuration
        """
        self.auth = auth
        self.base_url = auth.base_url
        self.timeout = ClientTimeout(total=auth.timeout)
        self._session: Optional[ClientSession] = None
        self._rate_limit_remaining: Optional[int] = None
        self._rate_limit_reset: Optional[datetime] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created."""
        if self._session is None or self._session.closed:
            headers = {
                'Authorization': f'token {self.auth.token}',
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': 'AgentWorld-GitHub-Client/1.0',
            }
            self._session = ClientSession(
                headers=headers,
                timeout=self.timeout,
                connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
            )
    
    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        if endpoint.startswith('http'):
            return endpoint
        return urljoin(self.base_url, endpoint.lstrip('/'))
    
    def _parse_rate_limit_headers(self, headers: Dict[str, str]) -> None:
        """Parse rate limit headers from response."""
        if 'X-RateLimit-Remaining' in headers:
            try:
                self._rate_limit_remaining = int(headers['X-RateLimit-Remaining'])
            except (ValueError, TypeError):
                pass
        
        if 'X-RateLimit-Reset' in headers:
            try:
                reset_timestamp = int(headers['X-RateLimit-Reset'])
                self._rate_limit_reset = datetime.fromtimestamp(reset_timestamp, tz=timezone.utc)
            except (ValueError, TypeError):
                pass
    
    def _handle_response_error(self, status_code: int, response_data: Any, headers: Dict[str, str]) -> None:
        """Handle HTTP response errors."""
        self._parse_rate_limit_headers(headers)
        
        # Check rate limit
        if status_code == 403 and 'rate limit' in str(response_data).lower():
            reset_time = self._rate_limit_reset
            raise RateLimitError(
                f"Rate limit exceeded. Reset at: {reset_time}",
                status_code=status_code,
                rate_limit_reset=reset_time
            )
        
        # Handle specific status codes
        if status_code == 401:
            raise AuthenticationError("Invalid or expired token", status_code)
        elif status_code == 403:
            raise PermissionError("Insufficient permissions", status_code)
        elif status_code == 404:
            raise NotFoundError("Resource not found", status_code)
        elif status_code == 409:
            raise ConflictError("Resource conflict", status_code)
        elif status_code == 422:
            raise ValidationError("Validation failed", status_code)
        elif status_code >= 500:
            raise APIError(f"GitHub API error: {status_code}", status_code)
        else:
            raise APIError(f"HTTP {status_code}: {response_data}", status_code)
    
    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> GitHubResponse:
        """Make HTTP request to GitHub API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            headers: Additional headers
            timeout: Request timeout
            
        Returns:
            GitHub API response
            
        Raises:
            Various GitHub exceptions based on response
        """
        await self._ensure_session()
        
        url = self._build_url(endpoint)
        request_headers = headers or {}
        
        # Prepare request data
        json_data = None
        if data is not None:
            json_data = data
            request_headers['Content-Type'] = 'application/json'
        
        # Use custom timeout if provided
        request_timeout = ClientTimeout(total=timeout) if timeout else self.timeout
        
        try:
            async with self._session.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=request_headers,
                timeout=request_timeout
            ) as response:
                # Parse response headers
                response_headers = dict(response.headers)
                self._parse_rate_limit_headers(response_headers)
                
                # Read response data
                try:
                    response_data = await response.json()
                except (json.JSONDecodeError, aiohttp.ContentTypeError):
                    response_data = await response.text()
                
                # Handle errors
                if response.status >= 400:
                    self._handle_response_error(response.status, response_data, response_headers)
                
                return GitHubResponse(
                    status_code=response.status,
                    headers=response_headers,
                    data=response_data,
                    rate_limit_remaining=self._rate_limit_remaining,
                    rate_limit_reset=self._rate_limit_reset
                )
        
        except asyncio.TimeoutError:
            raise TimeoutError("Request timeout")
        except aiohttp.ClientError as e:
            raise NetworkError(f"Network error: {str(e)}")
        except Exception as e:
            raise GitHubError(f"Unexpected error: {str(e)}")
    
    async def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> GitHubResponse:
        """Make GET request."""
        return await self.request('GET', endpoint, params=params, **kwargs)
    
    async def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> GitHubResponse:
        """Make POST request."""
        return await self.request('POST', endpoint, data=data, **kwargs)
    
    async def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> GitHubResponse:
        """Make PUT request."""
        return await self.request('PUT', endpoint, data=data, **kwargs)
    
    async def patch(self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> GitHubResponse:
        """Make PATCH request."""
        return await self.request('PATCH', endpoint, data=data, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> GitHubResponse:
        """Make DELETE request."""
        return await self.request('DELETE', endpoint, **kwargs)
    
    async def paginate(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: Optional[int] = None,
        per_page: int = 100
    ) -> List[Any]:
        """Paginate through GitHub API results.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            max_pages: Maximum number of pages to fetch
            per_page: Items per page
            
        Returns:
            List of all items from all pages
        """
        if params is None:
            params = {}
        
        params['per_page'] = min(per_page, 100)  # GitHub max is 100
        params['page'] = 1
        
        all_items = []
        page_count = 0
        
        while True:
            if max_pages and page_count >= max_pages:
                break
            
            response = await self.get(endpoint, params=params)
            items = response.data if isinstance(response.data, list) else []
            
            if not items:
                break
            
            all_items.extend(items)
            page_count += 1
            params['page'] += 1
            
            # Check if we've reached the last page
            if len(items) < params['per_page']:
                break
        
        return all_items
    
    @property
    def rate_limit_remaining(self) -> Optional[int]:
        """Get remaining rate limit."""
        return self._rate_limit_remaining
    
    @property
    def rate_limit_reset(self) -> Optional[datetime]:
        """Get rate limit reset time."""
        return self._rate_limit_reset
