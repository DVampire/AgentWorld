import re
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple


def truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars] + "\n... [truncated]", True


def run_command(
    cmd: List[str],
    timeout_sec: Optional[int] = None,
) -> Tuple[int, str, str, bool, float]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return proc.returncode, proc.stdout, proc.stderr, False, round(time.time() - start, 3)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        timeout_msg = f"\n[timeout] Command exceeded {timeout_sec}s."
        return 124, stdout, (stderr + timeout_msg).strip(), True, round(time.time() - start, 3)


def sanitize_tag_part(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9_.-]+", "-", value)
    value = value.strip("-")
    return value or "task"


def docker_image_exists(tag: str) -> bool:
    rc, _, _, _, _ = run_command(["docker", "image", "inspect", tag], timeout_sec=30)
    return rc == 0


def read_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_env_map(env_vars: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not env_vars:
        return {}
    return {str(k): str(v) for k, v in env_vars.items()}


def docker_cleanup_all(remove_images: bool = True) -> None:
    """Force-remove all skillsbench sandbox containers and images.

    Safe to call multiple times; silently ignores errors.
    """
    # 1. Remove all containers matching the naming convention
    rc, stdout, _, _, _ = run_command(
        ["docker", "ps", "-a", "--filter", "name=skillsbench-sbx-", "--format", "{{.ID}}"],
        timeout_sec=30,
    )
    if rc == 0 and stdout.strip():
        container_ids = stdout.strip().split()
        run_command(["docker", "rm", "-f", *container_ids], timeout_sec=60)

    if not remove_images:
        return

    # 2. Remove all images matching the naming convention
    #    Use positional repo glob (more reliable than --filter reference=)
    rc, stdout, _, _, _ = run_command(
        ["docker", "images", "skillsbench-sandbox-*", "--format", "{{.ID}}"],
        timeout_sec=30,
    )
    if rc == 0 and stdout.strip():
        # Deduplicate IDs (same image may appear for multiple tags)
        image_ids = list(dict.fromkeys(stdout.strip().split()))
        run_command(["docker", "rmi", "-f", *image_ids], timeout_sec=120)
