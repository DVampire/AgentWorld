import argparse
import signal
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .config import SUPPORTED_DATASETS
from .http_api import SandboxHTTPServer
from .manager import SandboxManager
from .process import read_int

DEFAULT_CONFIG_FILENAME = "sandbox_config.yaml"
DEFAULT_CONFIG_FALLBACK = Path("configs") / DEFAULT_CONFIG_FILENAME


def default_repo_root() -> Path:
    project_root = Path(__file__).resolve().parents[1]
    bundled_repo = project_root / "skillsbench"
    if bundled_repo.exists():
        return bundled_repo
    return project_root


def _auto_config_path() -> Optional[Path]:
    candidate = Path.cwd() / DEFAULT_CONFIG_FILENAME
    if candidate.exists():
        return candidate.resolve()
    fallback = Path.cwd() / DEFAULT_CONFIG_FALLBACK
    if fallback.exists():
        return fallback.resolve()
    return None


def _resolve_config_path(raw_path: Optional[Path]) -> Optional[Path]:
    if raw_path is None:
        return _auto_config_path()
    resolved = raw_path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Config file not found: {resolved}")
    return resolved


def _strip_inline_comment(value: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(value):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return value[:idx].strip()
    return value.strip()


def _parse_yaml_scalar(raw_value: str) -> Any:
    value = _strip_inline_comment(raw_value).strip()
    if value == "" or value.lower() in {"null", "~"}:
        return None
    lower = value.lower()
    if lower in {"true", "yes"}:
        return True
    if lower in {"false", "no"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if raw_line.startswith(" ") or raw_line.startswith("\t"):
            raise ValueError(
                "Nested YAML requires PyYAML. Install PyYAML or use flat key:value config."
            )

        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        line = _strip_inline_comment(line)
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid YAML line {line_no} in {path}: {raw_line}")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid YAML key at line {line_no} in {path}")
        data[key] = _parse_yaml_scalar(raw_value)

    return data


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        yaml = None

    if yaml is not None:
        with path.open("r") as f:
            config = yaml.safe_load(f)
        if config is None:
            return {}
        if not isinstance(config, dict):
            raise ValueError(f"YAML config must be a mapping: {path}")
        return config

    return _load_simple_yaml(path)


def _resolve_path_value(raw_value: Any, config_path: Optional[Path]) -> Path:
    path = Path(str(raw_value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    if config_path is not None:
        return (config_path.parent / path).resolve()
    return path.resolve()


def _build_runtime_overrides(config: Dict[str, Any], config_path: Optional[Path]) -> Dict[str, Any]:
    runtime_keys = {
        "default_step_timeout_sec",
        "default_eval_timeout_sec",
        "default_workdir",
        "max_timeout_sec",
        "max_output_chars_default",
        "run_dir_base",
    }
    overrides: Dict[str, Any] = {}
    for key in runtime_keys:
        value = config.get(key)
        if value is None:
            continue
        if key == "run_dir_base":
            overrides[key] = str(_resolve_path_value(value, config_path))
        else:
            overrides[key] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SkillsBench RL Sandbox HTTP Server")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML config path (default auto-load: ./{DEFAULT_CONFIG_FILENAME} if exists)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind (CLI overrides YAML/default)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (CLI overrides YAML/default)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="SkillsBench repository root (CLI overrides YAML/default)",
    )
    parser.add_argument(
        "--default-dataset",
        default=None,
        choices=SUPPORTED_DATASETS,
        help="Default dataset when POST /envs omits dataset (CLI overrides YAML/default)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = _resolve_config_path(args.config)
    file_config = _load_yaml_config(config_path) if config_path else {}

    host = args.host if args.host else str(file_config.get("host", "0.0.0.0"))
    port = args.port if args.port is not None else read_int(file_config.get("port"), 8080)
    if port < 1 or port > 65535:
        raise ValueError(f"Invalid port: {port}")

    if args.repo_root is not None:
        repo_root = args.repo_root.expanduser().resolve()
    elif file_config.get("repo_root") is not None:
        repo_root = _resolve_path_value(file_config.get("repo_root"), config_path)
    else:
        repo_root = default_repo_root()

    default_dataset = args.default_dataset if args.default_dataset else str(file_config.get("default_dataset", "tasks"))
    if default_dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            "default_dataset must be one of: {}".format(", ".join(SUPPORTED_DATASETS))
        )

    runtime_overrides = _build_runtime_overrides(file_config, config_path)
    manager = SandboxManager(
        repo_root=repo_root,
        default_dataset=default_dataset,
        runtime_overrides=runtime_overrides,
    )
    server = SandboxHTTPServer((host, port), manager)

    shutdown_event = threading.Event()

    def _shutdown_handler(signum: int, frame: Any) -> None:
        if shutdown_event.is_set():
            return
        shutdown_event.set()
        print(f"\nReceived signal {signum}; shutting down sandbox server...")
        server.shutdown()

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    print(f"SkillsBench sandbox server listening on http://{host}:{port}")
    print(f"Repo root: {repo_root.resolve()}")
    print(f"Default dataset: {default_dataset}")
    if config_path is not None:
        print(f"Config file: {config_path}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    finally:
        manager.shutdown(remove_images=True)
        server.server_close()
