import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    DEFAULT_EVAL_TIMEOUT_SEC,
    DEFAULT_STEP_TIMEOUT_SEC,
    DEFAULT_WORKDIR,
    MAX_OUTPUT_CHARS_DEFAULT,
    MAX_TIMEOUT_SEC,
    RUN_DIR_BASE,
    SUPPORTED_DATASETS,
)
from .errors import SandboxError
from .models import SandboxEnv, TaskInfo
from .process import as_env_map, docker_image_exists, read_int, run_command, sanitize_tag_part, truncate
from .task_config import load_task_toml


class SandboxManager:
    def __init__(
        self,
        repo_root: Path,
        default_dataset: str = "tasks",
        runtime_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.repo_root = repo_root.resolve()
        self.default_dataset = default_dataset
        self._lock = threading.RLock()
        self._envs: Dict[str, SandboxEnv] = {}

        runtime_overrides = runtime_overrides or {}
        self.default_step_timeout_sec = max(
            1, read_int(runtime_overrides.get("default_step_timeout_sec"), DEFAULT_STEP_TIMEOUT_SEC)
        )
        self.default_eval_timeout_sec = max(
            1, read_int(runtime_overrides.get("default_eval_timeout_sec"), DEFAULT_EVAL_TIMEOUT_SEC)
        )
        self.default_workdir = str(runtime_overrides.get("default_workdir", DEFAULT_WORKDIR))
        self.max_timeout_sec = max(1, read_int(runtime_overrides.get("max_timeout_sec"), MAX_TIMEOUT_SEC))
        self.max_output_chars_default = max(
            1024, read_int(runtime_overrides.get("max_output_chars_default"), MAX_OUTPUT_CHARS_DEFAULT)
        )
        run_dir_base_raw = runtime_overrides.get("run_dir_base", RUN_DIR_BASE)
        self.run_dir_base = Path(str(run_dir_base_raw)).resolve()
        self.run_dir_base.mkdir(parents=True, exist_ok=True)

    def _resolve_task(self, payload: Dict[str, Any]) -> TaskInfo:
        task_path_raw = payload.get("task_path")
        dataset = str(payload.get("dataset", self.default_dataset))

        if task_path_raw:
            task_path = Path(task_path_raw)
            if not task_path.is_absolute():
                task_path = (self.repo_root / task_path).resolve()
            dataset = task_path.parent.name
            task_id = task_path.name
        else:
            task_id = str(payload.get("task_id", "")).strip()
            if not task_id:
                raise SandboxError("task_id is required when task_path is not provided")
            if dataset not in SUPPORTED_DATASETS:
                raise SandboxError(f"Unsupported dataset '{dataset}'. Supported: {SUPPORTED_DATASETS}")
            task_path = (self.repo_root / dataset / task_id).resolve()

        if not task_path.exists():
            raise SandboxError(f"Task path does not exist: {task_path}", 404)

        environment_path = task_path / "environment"
        dockerfile_path = environment_path / "Dockerfile"
        tests_path = task_path / "tests"
        instruction_path = task_path / "instruction.md"
        if not dockerfile_path.exists():
            raise SandboxError(f"Dockerfile not found: {dockerfile_path}", 404)
        if not tests_path.exists():
            raise SandboxError(f"tests directory not found: {tests_path}", 404)
        if not instruction_path.exists():
            raise SandboxError(f"instruction.md not found: {instruction_path}", 404)

        config = load_task_toml(task_path)
        env_cfg = config.get("environment", {}) if isinstance(config.get("environment"), dict) else {}
        agent_cfg = config.get("agent", {}) if isinstance(config.get("agent"), dict) else {}
        verifier_cfg = config.get("verifier", {}) if isinstance(config.get("verifier"), dict) else {}

        return TaskInfo(
            task_path=task_path,
            task_id=task_path.name,
            dataset=dataset,
            environment_path=environment_path,
            tests_path=tests_path,
            instruction_path=instruction_path,
            allow_internet=bool(env_cfg.get("allow_internet", True)),
            agent_timeout_sec=read_int(agent_cfg.get("timeout_sec"), 900),
            verifier_timeout_sec=read_int(verifier_cfg.get("timeout_sec"), 900),
            has_artifact=isinstance(config.get("artifact"), dict),
        )

    def _build_image(self, task_info: TaskInfo, image_tag: str, rebuild: bool) -> Dict[str, Any]:
        if not rebuild and docker_image_exists(image_tag):
            return {"built": False, "image_tag": image_tag}

        dockerfile = task_info.environment_path / "Dockerfile"
        cmd = [
            "docker",
            "build",
            "-t",
            image_tag,
            "-f",
            str(dockerfile),
            str(task_info.environment_path),
        ]
        rc, stdout, stderr, timed_out, duration = run_command(cmd, timeout_sec=self.max_timeout_sec)
        if rc != 0:
            detail = stderr if stderr else "(no stderr)"
            raise SandboxError(f"Failed to build task image: {detail}", 500)
        return {
            "built": True,
            "image_tag": image_tag,
            "build_duration_sec": duration,
            "build_stdout": stdout[-8000:],
            "build_stderr": stderr[-8000:],
            "build_timed_out": timed_out,
        }

    def _start_container(self, env: SandboxEnv) -> None:
        logs_mount = f"{env.logs_dir.resolve()}:/logs"
        tests_mount = f"{env.task_info.tests_path.resolve()}:/tests:ro"
        instruction_mount = f"{env.task_info.instruction_path.resolve()}:/instruction.md:ro"

        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            env.container_name,
            "-v",
            logs_mount,
            "-v",
            tests_mount,
            "-v",
            instruction_mount,
            "-w",
            self.default_workdir,
        ]
        if not env.task_info.allow_internet:
            cmd.extend(["--network", "none"])
        cmd.extend([env.image_tag, "sh", "-lc", "while true; do sleep 3600; done"])

        rc, _, stderr, _, _ = run_command(cmd, timeout_sec=60)
        if rc != 0:
            raise SandboxError(f"Failed to start container: {stderr}", 500)

    def _stop_container(self, env: SandboxEnv) -> None:
        run_command(["docker", "rm", "-f", env.container_name], timeout_sec=30)

    def _exec(
        self,
        env: SandboxEnv,
        command: str,
        timeout_sec: int,
        workdir: Optional[str] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        effective_workdir = workdir if workdir else self.default_workdir
        cmd = ["docker", "exec", "-i", "-w", effective_workdir]
        if extra_env:
            for key, value in extra_env.items():
                cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([env.container_name, "sh", "-lc", command])

        rc, stdout, stderr, timed_out, duration = run_command(cmd, timeout_sec=timeout_sec)
        return {
            "exit_code": rc,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": timed_out,
            "duration_sec": duration,
            "command": command,
            "workdir": effective_workdir,
        }

    def create_env(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task_info = self._resolve_task(payload)
        if task_info.has_artifact:
            raise SandboxError("Tasks with [artifact] are not supported in this server yet.", 400)

        env_id = uuid.uuid4().hex[:12]
        image_part = sanitize_tag_part(f"{task_info.dataset}-{task_info.task_id}")
        image_hash = uuid.uuid5(uuid.NAMESPACE_URL, str(task_info.task_path)).hex[:10]
        image_tag = f"skillsbench-sandbox-{image_part}:{image_hash}"
        container_name = f"skillsbench-sbx-{env_id}"

        run_dir = self.run_dir_base / env_id
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        build_info = self._build_image(task_info, image_tag=image_tag, rebuild=bool(payload.get("rebuild", False)))
        env = SandboxEnv(
            env_id=env_id,
            task_info=task_info,
            image_tag=image_tag,
            container_name=container_name,
            run_dir=run_dir,
            logs_dir=logs_dir,
        )
        self._start_container(env)

        with self._lock:
            self._envs[env_id] = env

        return {
            "env_id": env_id,
            "task_id": task_info.task_id,
            "dataset": task_info.dataset,
            "task_path": str(task_info.task_path),
            "image_tag": image_tag,
            "container_name": container_name,
            "allow_internet": task_info.allow_internet,
            "instruction_path": str(task_info.instruction_path),
            "run_dir": str(run_dir),
            "build": build_info,
        }

    def _get_env(self, env_id: str) -> SandboxEnv:
        with self._lock:
            env = self._envs.get(env_id)
        if env is None:
            raise SandboxError(f"Unknown env_id: {env_id}", 404)
        return env

    def list_envs(self) -> List[Dict[str, Any]]:
        with self._lock:
            envs = list(self._envs.values())
        data: List[Dict[str, Any]] = []
        now = time.time()
        for env in envs:
            data.append(
                {
                    "env_id": env.env_id,
                    "task_id": env.task_info.task_id,
                    "dataset": env.task_info.dataset,
                    "created_at": env.created_at,
                    "age_sec": round(now - env.created_at, 3),
                    "steps": env.steps,
                    "last_reward": env.last_reward,
                    "container_name": env.container_name,
                }
            )
        return data

    def get_env(self, env_id: str) -> Dict[str, Any]:
        env = self._get_env(env_id)
        return {
            "env_id": env.env_id,
            "task_id": env.task_info.task_id,
            "dataset": env.task_info.dataset,
            "task_path": str(env.task_info.task_path),
            "image_tag": env.image_tag,
            "container_name": env.container_name,
            "allow_internet": env.task_info.allow_internet,
            "steps": env.steps,
            "last_reward": env.last_reward,
            "created_at": env.created_at,
            "instruction_path": str(env.instruction_path),
            "run_dir": str(env.run_dir),
            "logs_dir": str(env.logs_dir),
        }

    def get_instruction(self, env_id: str) -> Dict[str, Any]:
        env = self._get_env(env_id)
        text = env.instruction_path.read_text()
        return {
            "env_id": env_id,
            "task_id": env.task_info.task_id,
            "instruction": text,
        }

    def step(self, env_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        env = self._get_env(env_id)
        command = str(payload.get("command", "")).strip()
        if not command:
            raise SandboxError("Missing required field: command", 400)

        timeout_sec = read_int(payload.get("timeout_sec"), self.default_step_timeout_sec)
        timeout_sec = max(1, min(timeout_sec, self.max_timeout_sec))
        workdir = str(payload.get("workdir", self.default_workdir))
        max_output_chars = read_int(payload.get("max_output_chars"), self.max_output_chars_default)
        max_output_chars = max(1024, min(max_output_chars, 2_000_000))

        env_vars = payload.get("env")
        if env_vars is not None and not isinstance(env_vars, dict):
            raise SandboxError("env must be a key-value object", 400)
        safe_env = as_env_map(env_vars)

        with env.lock:
            result = self._exec(
                env=env,
                command=command,
                timeout_sec=timeout_sec,
                workdir=workdir,
                extra_env=safe_env,
            )
            env.steps += 1

        stdout, stdout_truncated = truncate(result["stdout"], max_output_chars)
        stderr, stderr_truncated = truncate(result["stderr"], max_output_chars)

        return {
            "env_id": env_id,
            "step": env.steps,
            "command": result["command"],
            "workdir": result["workdir"],
            "timeout_sec": timeout_sec,
            "duration_sec": result["duration_sec"],
            "timed_out": result["timed_out"],
            "exit_code": result["exit_code"],
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

    def evaluate(self, env_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        env = self._get_env(env_id)
        timeout_sec = read_int(payload.get("timeout_sec"), env.task_info.verifier_timeout_sec or self.default_eval_timeout_sec)
        timeout_sec = max(1, min(timeout_sec, self.max_timeout_sec))
        command = str(payload.get("command", "/bin/bash /tests/test.sh"))
        max_output_chars = read_int(payload.get("max_output_chars"), self.max_output_chars_default)
        max_output_chars = max(1024, min(max_output_chars, 2_000_000))

        verifier_dir = env.logs_dir / "verifier"
        if verifier_dir.exists():
            shutil.rmtree(verifier_dir, ignore_errors=True)
        verifier_dir.mkdir(parents=True, exist_ok=True)

        with env.lock:
            result = self._exec(
                env=env,
                command=f"cd {self.default_workdir} && {command}",
                timeout_sec=timeout_sec,
                workdir=self.default_workdir,
            )

        reward_file = verifier_dir / "reward.txt"
        reward: Optional[float] = None
        if reward_file.exists():
            raw = reward_file.read_text().strip()
            if raw:
                try:
                    reward = float(raw.split()[0])
                except ValueError:
                    reward = None

        ctrf_file = verifier_dir / "ctrf.json"
        env.last_reward = reward

        stdout, stdout_truncated = truncate(result["stdout"], max_output_chars)
        stderr, stderr_truncated = truncate(result["stderr"], max_output_chars)

        return {
            "env_id": env_id,
            "command": result["command"],
            "duration_sec": result["duration_sec"],
            "timeout_sec": timeout_sec,
            "timed_out": result["timed_out"],
            "exit_code": result["exit_code"],
            "reward": reward,
            "done": reward == 1.0 if reward is not None else False,
            "reward_file": str(reward_file),
            "ctrf_file": str(ctrf_file) if ctrf_file.exists() else None,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": stdout_truncated,
            "stderr_truncated": stderr_truncated,
        }

    def reset(self, env_id: str) -> Dict[str, Any]:
        env = self._get_env(env_id)
        with env.lock:
            self._stop_container(env)
            self._start_container(env)
            env.steps = 0
            env.last_reward = None

            verifier_dir = env.logs_dir / "verifier"
            if verifier_dir.exists():
                shutil.rmtree(verifier_dir, ignore_errors=True)
            verifier_dir.mkdir(parents=True, exist_ok=True)

        return {
            "env_id": env_id,
            "reset": True,
            "container_name": env.container_name,
            "steps": env.steps,
        }

    def delete_env(self, env_id: str, remove_image: bool = False) -> Dict[str, Any]:
        with self._lock:
            env = self._envs.pop(env_id, None)
        if env is None:
            raise SandboxError(f"Unknown env_id: {env_id}", 404)

        with env.lock:
            self._stop_container(env)
            shutil.rmtree(env.run_dir, ignore_errors=True)
            if remove_image:
                run_command(["docker", "image", "rm", env.image_tag], timeout_sec=120)

        return {
            "env_id": env_id,
            "deleted": True,
            "image_removed": remove_image,
        }

    def remove_image(self, image_tag: str) -> Dict[str, Any]:
        """Remove a Docker image by tag, independent of any env lifecycle."""
        exit_code, stdout, stderr, timed_out, _ = run_command(
            ["docker", "image", "rm", image_tag], timeout_sec=120
        )
        if exit_code != 0:
            raise SandboxError(
                f"Failed to remove image {image_tag}: {stderr.strip()}", 500
            )
        return {"image_tag": image_tag, "removed": True}

    def shutdown(self, remove_images: bool = True) -> None:
        with self._lock:
            env_ids = list(self._envs.keys())
        for env_id in env_ids:
            try:
                self.delete_env(env_id, remove_image=remove_images)
            except Exception:
                pass
