from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AsyncLockManager:
    """Per-key asyncio locks for safe concurrent file operations."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._master = asyncio.Lock()

    async def _get_lock(self, key: str) -> asyncio.Lock:
        async with self._master:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    @asynccontextmanager
    async def acquire(self, key: str) -> AsyncIterator[None]:
        lock = await self._get_lock(key)
        await lock.acquire()
        try:
            yield None
        finally:
            lock.release()


