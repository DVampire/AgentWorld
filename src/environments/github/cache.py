"""Cache implementation for GitHub service."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional, Tuple
from collections import OrderedDict


class LRUCache:
    """Thread-safe LRU cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time to live for cached items in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                return None
            
            value, timestamp = self._cache[key]
            
            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value
    
    async def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            # Remove if exists
            if key in self._cache:
                del self._cache[key]
            
            # Add new item
            self._cache[key] = (value, time.time())
            
            # Remove oldest if over limit
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)
    
    async def delete(self, key: str) -> None:
        """Delete item from cache.
        
        Args:
            key: Cache key to delete
        """
        async with self._lock:
            self._cache.pop(key, None)
    
    async def clear(self) -> None:
        """Clear all items from cache."""
        async with self._lock:
            self._cache.clear()
    
    async def size(self) -> int:
        """Get current cache size."""
        async with self._lock:
            return len(self._cache)
    
    async def cleanup_expired(self) -> int:
        """Remove expired items from cache.
        
        Returns:
            Number of items removed
        """
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self.ttl_seconds
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)


class GitHubCache:
    """GitHub-specific cache with different TTL for different data types."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize GitHub cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        # Different TTL for different data types
        self._user_cache = LRUCache(max_size, ttl_seconds=3600)  # 1 hour
        self._repo_cache = LRUCache(max_size, ttl_seconds=1800)  # 30 minutes
        self._content_cache = LRUCache(max_size, ttl_seconds=300)  # 5 minutes
        self._search_cache = LRUCache(max_size, ttl_seconds=600)  # 10 minutes
        self._default_cache = LRUCache(max_size, ttl_seconds=300)  # 5 minutes
    
    def _get_cache(self, cache_type: str) -> LRUCache:
        """Get appropriate cache based on type."""
        cache_map = {
            'user': self._user_cache,
            'repo': self._repo_cache,
            'content': self._content_cache,
            'search': self._search_cache,
        }
        return cache_map.get(cache_type, self._default_cache)
    
    async def get(self, key: str, cache_type: str = 'default') -> Optional[Any]:
        """Get item from appropriate cache."""
        cache = self._get_cache(cache_type)
        return await cache.get(key)
    
    async def put(self, key: str, value: Any, cache_type: str = 'default') -> None:
        """Put item in appropriate cache."""
        cache = self._get_cache(cache_type)
        await cache.put(key, value)
    
    async def delete(self, key: str, cache_type: str = 'default') -> None:
        """Delete item from appropriate cache."""
        cache = self._get_cache(cache_type)
        await cache.delete(key)
    
    async def clear(self, cache_type: Optional[str] = None) -> None:
        """Clear cache(s)."""
        if cache_type:
            cache = self._get_cache(cache_type)
            await cache.clear()
        else:
            # Clear all caches
            for cache in [self._user_cache, self._repo_cache, self._content_cache, 
                         self._search_cache, self._default_cache]:
                await cache.clear()
    
    async def cleanup_expired(self) -> int:
        """Cleanup expired items from all caches."""
        total_removed = 0
        for cache in [self._user_cache, self._repo_cache, self._content_cache, 
                     self._search_cache, self._default_cache]:
            removed = await cache.cleanup_expired()
            total_removed += removed
        return total_removed
