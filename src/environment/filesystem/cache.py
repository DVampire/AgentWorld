from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Union
import threading


@dataclass
class CacheEntry:
    """Cache entry with data, timestamp, and access count."""
    data: bytes
    timestamp: float
    access_count: int = 0
    size: int = 0
    
    def __post_init__(self):
        if self.size == 0:
            self.size = len(self.data)


class LRUByteCache:
    """Enhanced LRU cache for byte content with better memory management."""

    def __init__(self, max_entries: int = 256, max_bytes_total: int = 64 * 1024 * 1024, ttl_seconds: int = 3600) -> None:
        """Initialize the cache with configurable limits.
        
        Args:
            max_entries: Maximum number of cache entries
            max_bytes_total: Maximum total bytes in cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._max_entries = max_entries
        self._max_bytes_total = max_bytes_total
        self._ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._total_bytes = 0
        self._lock = threading.RLock()  # Thread-safe operations
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[bytes]:
        """Get cached data with TTL check and access tracking."""
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                self._misses += 1
                return None
            
            # Check TTL
            if time.time() - entry.timestamp > self._ttl_seconds:
                self._delete_entry(key, entry)
                self._misses += 1
                return None
            
            # Update access count and move to end (recently used)
            entry.access_count += 1
            self._entries.move_to_end(key)
            self._hits += 1
            return entry.data

    def put(self, key: str, data: bytes) -> None:
        """Put data in cache with size tracking."""
        with self._lock:
            data_size = len(data)
            
            # Remove existing entry if present
            if key in self._entries:
                old_entry = self._entries.pop(key)
                self._total_bytes -= old_entry.size
            
            # Create new entry
            entry = CacheEntry(data=data, timestamp=time.time(), size=data_size)
            self._entries[key] = entry
            self._total_bytes += data_size
            
            self._evict_if_needed()

    def delete(self, key: str) -> None:
        """Delete entry from cache."""
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._total_bytes -= entry.size

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._entries.clear()
            self._total_bytes = 0
            self._hits = 0
            self._misses = 0

    def _delete_entry(self, key: str, entry: CacheEntry) -> None:
        """Delete entry and update total bytes."""
        self._entries.pop(key, None)
        self._total_bytes -= entry.size

    def _evict_if_needed(self) -> None:
        """Evict entries based on size and count limits."""
        current_time = time.time()
        
        # First, remove expired entries
        expired_keys = [
            key for key, entry in self._entries.items()
            if current_time - entry.timestamp > self._ttl_seconds
        ]
        for key in expired_keys:
            self._delete_entry(key, self._entries[key])
        
        # Then evict by entry count
        while len(self._entries) > self._max_entries:
            _, old_entry = self._entries.popitem(last=False)
            self._total_bytes -= old_entry.size
        
        # Finally evict by total size (evict least recently used)
        while self._total_bytes > self._max_bytes_total and self._entries:
            _, old_entry = self._entries.popitem(last=False)
            self._total_bytes -= old_entry.size

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            return {
                'entries': len(self._entries),
                'total_bytes': self._total_bytes,
                'max_entries': self._max_entries,
                'max_bytes': self._max_bytes_total,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'ttl_seconds': self._ttl_seconds
            }


