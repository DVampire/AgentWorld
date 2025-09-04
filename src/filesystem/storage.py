from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Protocol


class StorageBackend(Protocol):
    async def read_bytes(self, path: Path) -> bytes: ...

    async def write_bytes(self, path: Path, data: bytes, *, overwrite: bool) -> None: ...

    async def stat(self, path: Path) -> os.stat_result: ...

    async def listdir(self, path: Path) -> list[str]: ...

    async def exists(self, path: Path) -> bool: ...

    async def mkdir(self, path: Path, *, parents: bool) -> None: ...

    async def remove(self, path: Path) -> None: ...

    async def rename(self, old: Path, new: Path) -> None: ...

    async def rmtree(self, path: Path) -> None: ...


class LocalAsyncStorage:
    """Async adapters over local filesystem operations using threads."""

    async def read_bytes(self, path: Path) -> bytes:
        return await asyncio.to_thread(path.read_bytes)

    async def write_bytes(self, path: Path, data: bytes, *, overwrite: bool) -> None:
        def _write() -> None:
            if not overwrite and path.exists():
                raise FileExistsError(str(path))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        await asyncio.to_thread(_write)

    async def stat(self, path: Path) -> os.stat_result:
        return await asyncio.to_thread(path.stat)

    async def listdir(self, path: Path) -> list[str]:
        def _listdir():
            if path.is_absolute():
                return [p.name for p in path.iterdir()]
            else:
                # For relative paths, we need to resolve them properly
                # This should not happen in normal usage as paths should be absolute
                return [p.name for p in path.iterdir()]
        return await asyncio.to_thread(_listdir)

    async def exists(self, path: Path) -> bool:
        return await asyncio.to_thread(path.exists)

    async def mkdir(self, path: Path, *, parents: bool) -> None:
        await asyncio.to_thread(lambda: path.mkdir(parents=parents, exist_ok=True))

    async def remove(self, path: Path) -> None:
        await asyncio.to_thread(lambda: path.unlink(missing_ok=False))

    async def rename(self, old: Path, new: Path) -> None:
        def _rename() -> None:
            new.parent.mkdir(parents=True, exist_ok=True)
            old.rename(new)

        await asyncio.to_thread(_rename)

    async def rmtree(self, path: Path) -> None:
        import shutil

        await asyncio.to_thread(lambda: shutil.rmtree(path))


