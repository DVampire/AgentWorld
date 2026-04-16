import os
from pathlib import Path

SUPPORTED_DATASETS = ("tasks", "tasks-no-skills", "tasks_no_skills_generate")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


DEFAULT_STEP_TIMEOUT_SEC = _env_int("SBX_DEFAULT_STEP_TIMEOUT_SEC", 60)
DEFAULT_EVAL_TIMEOUT_SEC = _env_int("SBX_DEFAULT_EVAL_TIMEOUT_SEC", 1800)
DEFAULT_WORKDIR = os.environ.get("SBX_DEFAULT_WORKDIR", "/root")
MAX_TIMEOUT_SEC = _env_int("SBX_MAX_TIMEOUT_SEC", 7200)
MAX_OUTPUT_CHARS_DEFAULT = _env_int("SBX_MAX_OUTPUT_CHARS_DEFAULT", 120_000)
RUN_DIR_BASE = Path(os.environ.get("SBX_RUN_DIR_BASE", "/tmp/skillsbench-sandbox"))
