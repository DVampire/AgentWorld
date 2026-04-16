from pathlib import Path
from typing import Any, Dict

from .errors import SandboxError

try:
    import tomllib as _toml
except ImportError:
    try:
        import tomli as _toml  # type: ignore[no-redef]
    except ImportError:
        _toml = None


def load_task_toml(task_path: Path) -> Dict[str, Any]:
    task_toml = task_path / "task.toml"
    if not task_toml.exists():
        raise SandboxError(f"task.toml not found in {task_path}", 404)
    if _toml is not None:
        with task_toml.open("rb") as f:
            return _toml.load(f)
    return _load_task_toml_fallback(task_toml)


def _parse_toml_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_task_toml_fallback(toml_file: Path) -> Dict[str, Any]:
    content = toml_file.read_text()
    config: Dict[str, Any] = {}
    current: Dict[str, Any] = config

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            parts = [part.strip() for part in section.split(".") if part.strip()]
            current = config
            for part in parts:
                if part not in current or not isinstance(current[part], dict):
                    current[part] = {}
                current = current[part]
            continue
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        current[key.strip()] = _parse_toml_value(raw_value)

    return config
