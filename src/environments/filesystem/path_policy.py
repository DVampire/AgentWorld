from __future__ import annotations

from pathlib import Path

from src.environments.filesystem.exceptions import InvalidPathError, PathTraversalError


class PathPolicy:
    """Sandboxed path resolution within a fixed base directory.

    All public APIs should accept relative paths. Internally we resolve to absolute
    paths under the configured base directory and guarantee no traversal escapes occur.
    """

    def __init__(self, base_dir: Path) -> None:
        if not isinstance(base_dir, Path):
            raise InvalidPathError("base_dir must be a pathlib.Path instance")
        self._base_dir = base_dir.resolve()

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def to_relative(self, path: Path) -> Path:
        absolute = path.resolve()
        try:
            return absolute.relative_to(self._base_dir)
        except ValueError as exc:
            raise PathTraversalError(
                f"Path '{absolute}' is outside of base_dir '{self._base_dir}'"
            ) from exc

    def resolve_relative(self, relative: str | Path) -> Path:
        if isinstance(relative, str):
            relative = Path(relative)
        if relative.is_absolute():
            raise InvalidPathError("Expected relative path under base_dir, got absolute path")
        absolute = (self._base_dir / relative).resolve()
        if not str(absolute).startswith(str(self._base_dir)):
            # Robust check against traversal
            raise PathTraversalError(
                f"Resolved path '{absolute}' escapes base_dir '{self._base_dir}'"
            )
        return absolute


