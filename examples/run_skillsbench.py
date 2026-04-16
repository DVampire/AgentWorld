"""Run SkillsBench benchmark — multi-turn agent interaction with sandbox environments.

SkillsBench tasks run inside Docker containers. The agent sends shell commands
and receives stdout/stderr feedback over multiple turns, then gets evaluated by
the sandbox's test harness.

Usage:
    # Direct LLM mode (model drives multi-turn shell interaction):
    python examples/run_skillsbench.py --model-name openrouter/gemini-3.1-pro-preview

    # Bus mode (full AgentOS pipeline — planner + sub-agents):
    python examples/run_skillsbench.py --use-bus

    # Single task:
    python examples/run_skillsbench.py --task-id adaptive-cruise-control --run-name test-bus-adapt-v2 --use-bus

    # Resume from latest results:
    python examples/run_skillsbench.py --resume

    # Filter: re-run only failed tasks:
    python examples/run_skillsbench.py --resume --filter wrong
"""

import asyncio
import json
import os
import re
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request

from dotenv import load_dotenv
load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.model import model_manager
from src.version import version_manager
from src.message.types import SystemMessage, HumanMessage, AssistantMessage

from skillsbench_tracer import SkillsBenchTracer


def _serialize_messages(messages) -> list:
    """Serialize a list of Message objects to plain dicts for trajectory logging."""
    out = []
    for m in messages:
        entry = {"role": m.role}
        if isinstance(m.content, str):
            entry["content"] = m.content
        else:
            entry["content"] = str(m.content)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# SkillsBench sandbox HTTP client (sync, thin wrapper)
# ---------------------------------------------------------------------------

SKILLSBENCH_PROJECT_ROOT = Path(root) / "src" / "benchmark" / "skillsbench-sandbox"
DONE_TOKEN = "<<DONE>>"


class SandboxHTTPClient:
    """Minimal HTTP client for the SkillsBench sandbox server."""

    def __init__(self, base_url: str, timeout_sec: int = 1800):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def _json_request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = self.base_url + path
        body = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        req = request.Request(url=url, data=body, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            err_raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {url} failed: {err_raw}")
        except error.URLError as exc:
            raise RuntimeError(f"Request to {url} failed: {exc}")

        if not raw:
            return {}
        return json.loads(raw)

    def health(self) -> Dict[str, Any]:
        return self._json_request("GET", "/health")

    def create_env(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._json_request("POST", "/envs", payload)

    def get_instruction(self, env_id: str) -> Dict[str, Any]:
        return self._json_request("GET", f"/envs/{env_id}/instruction")

    def step(self, env_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._json_request("POST", f"/envs/{env_id}/step", payload)

    def evaluate(self, env_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._json_request("POST", f"/envs/{env_id}/evaluate", payload or {})

    def reset_env(self, env_id: str) -> Dict[str, Any]:
        return self._json_request("POST", f"/envs/{env_id}/reset", {})

    def delete_env(self, env_id: str, remove_image: bool = False) -> Dict[str, Any]:
        query = "?remove_image=true" if remove_image else ""
        return self._json_request("DELETE", f"/envs/{env_id}{query}")

    def list_envs(self) -> Dict[str, Any]:
        return self._json_request("GET", "/envs")


def _is_server_healthy(base_url: str) -> bool:
    try:
        client = SandboxHTTPClient(base_url, timeout_sec=5)
        result = client.health()
        return str(result.get("status", "")).lower() == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Task metadata helpers (read from skillsbench repo on disk)
# ---------------------------------------------------------------------------

def _extract_frontmatter_value(text: str, key: str) -> str:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    end_index = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_index = i
            break
    if end_index < 0:
        return ""
    want = key.strip().lower()
    for i in range(1, end_index):
        line = lines[i]
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip().lower() == want:
            return v.strip().strip('"').strip("'")
    return ""


def _load_skill_metadata(task_dir: Path) -> List[Dict[str, str]]:
    skills_root = task_dir / "environment" / "skills"
    if not skills_root.is_dir():
        return []
    metadata = []
    for skill_dir in sorted(skills_root.iterdir(), key=lambda p: p.name):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        text = skill_md.read_text(errors="replace")
        skill_name = _extract_frontmatter_value(text, "name") or skill_dir.name
        description = _extract_frontmatter_value(text, "description") or "No description."
        metadata.append({
            "name": skill_name,
            "description": description,
            "path": f"/root/.codex/skills/{skill_dir.name}",
        })
    return metadata


def _extract_toml_section(text: str, section: str) -> str:
    pattern = r"(?ms)^\s*\[" + re.escape(section) + r"\]\s*(.*?)(?=^\s*\[|\Z)"
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def _extract_toml_string(text: str, section: str, key: str) -> Optional[str]:
    body = _extract_toml_section(text, section)
    if not body:
        return None
    pattern = r'(?m)^\s*' + re.escape(key) + r'\s*=\s*"([^"]+)"'
    match = re.search(pattern, body)
    return match.group(1).strip() if match else None


def _extract_toml_float(text: str, section: str, key: str) -> Optional[int]:
    body = _extract_toml_section(text, section)
    if not body:
        return None
    pattern = r"(?m)^\s*" + re.escape(key) + r"\s*=\s*([0-9]+(?:\.[0-9]+)?)"
    match = re.search(pattern, body)
    if not match:
        return None
    try:
        return int(float(match.group(1)))
    except Exception:
        return None


def _difficulty_to_max_steps(difficulty: Optional[str]) -> int:
    """Core=10, Extended=30, Extreme=50 (per SkillsBench paper)."""
    diff = str(difficulty or "").strip().lower()
    if diff in ("core", "easy"):
        return 10
    if diff in ("extreme", "hard"):
        return 50
    return 30  # medium / extended


def _load_task_meta(task_dir: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "task_dir": str(task_dir),
        "difficulty": "",
        "default_max_steps": 30,
        "agent_timeout_sec": 60,
        "verifier_timeout_sec": 600,
        "skills_metadata": [],
    }
    meta["skills_metadata"] = _load_skill_metadata(task_dir)

    task_toml = task_dir / "task.toml"
    if not task_toml.exists():
        return meta

    text = task_toml.read_text(errors="replace")
    difficulty = _extract_toml_string(text, "metadata", "difficulty")
    meta["difficulty"] = difficulty or ""
    meta["default_max_steps"] = _difficulty_to_max_steps(difficulty)
    agent_timeout = _extract_toml_float(text, "agent", "timeout_sec")
    if agent_timeout is not None:
        meta["agent_timeout_sec"] = agent_timeout
    verifier_timeout = _extract_toml_float(text, "verifier", "timeout_sec")
    if verifier_timeout is not None:
        meta["verifier_timeout_sec"] = verifier_timeout
    return meta


def _format_skills_section(skills: List[Dict[str, str]]) -> str:
    if not skills:
        return ""
    lines = [
        "## Available Skills",
        "",
        "You have access to the skills listed below. Proactively use any relevant skill.",
        "",
    ]
    for s in skills:
        lines.append(f"- **{s['name']}** (path: `{s['path']}`): {s['description']}")
    lines.append("\nTo use a skill, read its full instructions: `cat <path>/SKILL.md`")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Discover all tasks from the skillsbench repo
# ---------------------------------------------------------------------------

def discover_tasks(dataset: str, task_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Discover tasks from the skillsbench repo on disk."""
    tasks_dir = SKILLSBENCH_PROJECT_ROOT / "skillsbench" / dataset
    if not tasks_dir.is_dir():
        logger.error(f"| Tasks directory not found: {tasks_dir}")
        return []

    all_task_dirs = sorted([d for d in tasks_dir.iterdir() if d.is_dir()])
    if task_ids:
        all_task_dirs = [d for d in all_task_dirs if d.name in task_ids]

    tasks = []
    for task_dir in all_task_dirs:
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.exists():
            continue
        meta = _load_task_meta(task_dir)
        tasks.append({
            "task_id": task_dir.name,
            "dataset": dataset,
            "task_dir": task_dir,
            "instruction": instruction_path.read_text(errors="replace").strip(),
            "meta": meta,
        })
    return tasks


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run SkillsBench benchmark with AgentOS")
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs", "bus.py"),
        help="AgentOS config file path",
    )
    parser.add_argument(
        "--model-name", type=str, default="openrouter/gemini-3.1-pro-preview",
        help="model to use for direct LLM mode",
    )
    parser.add_argument(
        "--use-bus",
        action="store_true",
        default=False,
        help="use the AgentBus pipeline instead of direct LLM multi-turn",
    )
    parser.add_argument(
        "--server-url", type=str, default="http://127.0.0.1:8080",
        help="SkillsBench sandbox server URL",
    )
    parser.add_argument(
        "--dataset", type=str, default="tasks",
        help="SkillsBench dataset name (tasks, tasks-no-skills, etc.)",
    )
    parser.add_argument(
        "--task-id", type=str, default=None,
        help="run a single task by ID (e.g. xlsx-recover-data)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=4,
        help="maximum concurrent tasks (default: 4)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="override max steps per task (default: inferred from difficulty)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=100,
        help="maximum planner rounds per task, bus mode only (default: 100)",
    )
    parser.add_argument(
        "--step-timeout", type=int, default=None,
        help="override per-step execution timeout in seconds",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="start index of task subset (inclusive)",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="end index of task subset (exclusive)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from the latest results JSON, skipping completed tasks",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["wrong", "null"],
        default=None,
        help="requires --resume. 'wrong' re-runs failed tasks; 'null' re-runs tasks with no result.",
    )
    parser.add_argument(
        "--disable-skill-injection",
        action="store_true",
        default=False,
        help="disable injecting skill metadata into the agent prompt",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="custom name for this run (used in result filename and trajectory directory)",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        default=False,
        help="disable trajectory tracing (per-task JSON files with full interaction logs)",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=None,
        help="override config settings in xxx=yyy format",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Result saver
# ---------------------------------------------------------------------------

class BenchmarkResultSaver:
    """Save benchmark results to JSON with real-time updates."""

    def __init__(self, benchmark_name: str, concurrency: int, total_tasks: int, model_name: str, run_name: str = None):
        self.benchmark_name = benchmark_name
        self.start_time = datetime.now()

        results_dir = os.path.join(config.workdir, "results", benchmark_name)
        os.makedirs(results_dir, exist_ok=True)

        timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        if run_name:
            filename = f"{run_name}_{timestamp}.json"
        else:
            filename = f"benchmark_{benchmark_name}_{timestamp}.json"
        self.filepath = os.path.join(results_dir, filename)
        self.file_lock = asyncio.Lock()

        self.results_data = {
            "experiment_meta": {
                "timestamp": self.start_time.isoformat() + "Z",
                "benchmark": benchmark_name,
                "concurrency": concurrency,
                "total_tasks": total_tasks,
                "model": model_name,
            },
            "results": [],
            "summary": {
                "completed_tasks": 0,
                "total_reward": 0.0,
                "solved": 0,
                "accuracy": 0.0,
                "last_updated": self.start_time.isoformat() + "Z",
            },
        }

    def update_total_tasks(self, total_tasks: int) -> None:
        self.results_data["experiment_meta"]["total_tasks"] = total_tasks

    def preload_results(self, previous_results: list) -> None:
        self.results_data["results"] = sorted(previous_results, key=lambda r: r.get("task_id", ""))
        results = self.results_data["results"]
        total_reward = sum(r.get("reward", 0) for r in results)
        solved = sum(1 for r in results if r.get("reward", 0) == 1.0)
        self.results_data["summary"].update({
            "completed_tasks": len(results),
            "total_reward": total_reward,
            "solved": solved,
            "accuracy": solved / len(results) if results else 0.0,
            "last_updated": datetime.now().isoformat() + "Z",
        })
        self._flush()

    async def add_task_result(self, result: Dict[str, Any]) -> None:
        async with self.file_lock:
            self.results_data["results"].append(result)
            results = self.results_data["results"]
            total_reward = sum(r.get("reward", 0) for r in results)
            solved = sum(1 for r in results if r.get("reward", 0) == 1.0)
            self.results_data["summary"].update({
                "completed_tasks": len(results),
                "total_reward": total_reward,
                "solved": solved,
                "accuracy": solved / len(results) if results else 0.0,
                "last_updated": datetime.now().isoformat() + "Z",
            })
            self._flush()

    def _flush(self):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.results_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            logger.error(f"Failed to save results: {exc}")

    def get_file_path(self) -> str:
        return str(self.filepath)


# ---------------------------------------------------------------------------
# Direct LLM multi-turn interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous command-line agent operating inside a SkillsBench sandbox container.
Your objective is to solve the given task by executing shell commands.

Output contract:
- Return exactly ONE shell command per turn. No markdown fences, no explanation — just the raw command.
- When you believe the task is fully complete, output exactly: <<DONE>>
- Do NOT output <<DONE>> until you are confident the task is solved.

Execution policy:
- Follow the task instruction exactly, including file paths, names, schema, and units.
- Prefer small, verifiable steps; inspect before editing when uncertain.
- Preserve unrelated content and formatting in provided files.
"""


def _extract_command(text: str) -> str:
    """Extract a single shell command from LLM output."""
    text = text.strip()
    if not text:
        return ""
    if DONE_TOKEN in text:
        return DONE_TOKEN
    # Strip markdown code fences if the model wraps the command
    fence = re.search(r"```(?:bash|sh)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    # Strip "command:" prefix
    text = re.sub(r"^\s*command\s*:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


async def process_task_direct(
    task_info: Dict[str, Any],
    sandbox_client: SandboxHTTPClient,
    semaphore: asyncio.Semaphore,
    model_name: str,
    max_steps: Optional[int],
    step_timeout: Optional[int],
    include_skills: bool,
    result_saver: BenchmarkResultSaver,
    total_tasks: int,
    completed_count_ref: list,
    completed_lock: asyncio.Lock,
    tracer: Optional["SkillsBenchTracer"] = None,
) -> Dict[str, Any]:
    """Run one SkillsBench task with direct LLM multi-turn shell interaction."""
    task_id = task_info["task_id"]
    dataset = task_info["dataset"]
    meta = task_info["meta"]
    instruction = task_info["instruction"]

    effective_max_steps = max_steps or meta["default_max_steps"]
    effective_step_timeout = step_timeout or meta.get("agent_timeout_sec", 60)

    result = {
        "task_id": task_id,
        "dataset": dataset,
        "difficulty": meta.get("difficulty", ""),
        "reward": 0.0,
        "done": False,
        "steps": 0,
        "processing_time": 0.0,
        "commands": [],
    }

    # Begin trajectory tracing
    if tracer:
        await tracer.begin_task(task_id, dataset, meta.get("difficulty", ""), meta)

    async with semaphore:
        start_time = time.time()
        env_id = None
        task_error: Optional[str] = None
        try:
            logger.info(f"| {'='*50}")
            logger.info(f"| Processing Task (direct): {task_id}")
            logger.info(f"| {'='*50}")

            # 1. Create sandbox environment
            create_resp = await asyncio.to_thread(
                sandbox_client.create_env,
                {"dataset": dataset, "task_id": task_id},
            )
            env_id = create_resp["env_id"]
            logger.info(f"| Created env: {env_id}")

            # 2. Get instruction from sandbox (authoritative)
            instr_resp = await asyncio.to_thread(sandbox_client.get_instruction, env_id)
            sandbox_instruction = instr_resp.get("instruction", instruction)

            # 3. Build first-turn prompt
            first_turn_parts = [f"## Task Instruction\n\n{sandbox_instruction}"]
            if include_skills and meta.get("skills_metadata"):
                skills_section = _format_skills_section(meta["skills_metadata"])
                if skills_section:
                    first_turn_parts.append(skills_section)
            first_turn_parts.append("Begin. Return the first shell command.")
            first_turn_prompt = "\n\n".join(first_turn_parts)

            # 4. Multi-turn loop
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=first_turn_prompt),
            ]

            for step_idx in range(effective_max_steps):
                # Call LLM
                response = await model_manager(model=model_name, messages=messages)
                llm_text = response.message.strip() if response.message else ""
                command = _extract_command(llm_text)

                if not command:
                    logger.warning(f"| [{task_id}] Step {step_idx+1}: empty command, retrying")
                    messages.append(AssistantMessage(content=llm_text))
                    messages.append(HumanMessage(content="Error: empty command. Please provide a valid shell command."))
                    if tracer:
                        await tracer.record_step(
                            task_id, step_idx + 1,
                            llm_output=llm_text, command="",
                            note="empty command, retrying",
                        )
                    continue

                # Agent signals done
                if command == DONE_TOKEN:
                    logger.info(f"| [{task_id}] Step {step_idx+1}: agent signaled DONE")
                    result["steps"] = step_idx + 1
                    if tracer:
                        await tracer.record_step(
                            task_id, step_idx + 1,
                            llm_output=llm_text, command=DONE_TOKEN, is_done=True,
                        )
                    break

                # Execute command in sandbox
                logger.info(f"| [{task_id}] Step {step_idx+1}: {command[:120]}")
                step_resp = await asyncio.to_thread(
                    sandbox_client.step,
                    env_id,
                    {"command": command, "timeout_sec": effective_step_timeout},
                )

                exit_code = step_resp.get("exit_code", -1)
                stdout = step_resp.get("stdout", "")
                stderr = step_resp.get("stderr", "")
                result["commands"].append({
                    "step": step_idx + 1,
                    "command": command,
                    "exit_code": exit_code,
                })

                # Build feedback for next turn
                feedback = f"exit_code: {exit_code}\nstdout:\n{stdout}\nstderr:\n{stderr}"
                messages.append(AssistantMessage(content=llm_text))
                messages.append(HumanMessage(content=feedback))

                # Trace this step
                if tracer:
                    await tracer.record_step(
                        task_id, step_idx + 1,
                        prompt=_serialize_messages(messages),
                        llm_output=llm_text,
                        command=command,
                        exit_code=exit_code,
                        stdout=stdout,
                        stderr=stderr,
                    )

                result["steps"] = step_idx + 1

                # Sliding window: keep system + first user + last N turns
                max_context_turns = 40  # user+assistant pairs
                pinned = 2  # system + first user
                droppable = messages[pinned:]
                if len(droppable) > max_context_turns * 2:
                    messages = messages[:pinned] + droppable[-(max_context_turns * 2):]

            # 5. Evaluate
            logger.info(f"| [{task_id}] Evaluating...")
            eval_resp = await asyncio.to_thread(
                sandbox_client.evaluate,
                env_id,
                {"timeout_sec": meta.get("verifier_timeout_sec", 600)},
            )
            raw_reward = eval_resp.get("reward")
            reward = float(raw_reward) if raw_reward is not None else 0.0
            done = bool(eval_resp.get("done", False))
            result["reward"] = reward
            result["done"] = done

            tag = "✅ Solved" if done else f"❌ reward={reward}"
            logger.info(f"| {tag} [{task_id}]")

        except Exception as exc:
            logger.error(f"| [{task_id}] Error: {exc}")
            task_error = str(exc)

        finally:
            result["processing_time"] = time.time() - start_time

            # Cleanup sandbox env
            if env_id:
                try:
                    await asyncio.to_thread(sandbox_client.delete_env, env_id, False)
                except Exception:
                    pass

            await result_saver.add_task_result(result)

            # Finalize trajectory and flush to disk
            if tracer:
                await tracer.end_task(
                    task_id,
                    reward=result["reward"],
                    done=result["done"],
                    total_steps=result["steps"],
                    processing_time=result["processing_time"],
                    error=task_error,
                )

            async with completed_lock:
                completed_count_ref[0] += 1
                done_count = completed_count_ref[0]
                pct = done_count / total_tasks * 100
                logger.info(f"| Progress: {done_count}/{total_tasks} ({pct:.1f}%)")

    return result


# ---------------------------------------------------------------------------
# Bus-based interaction
# ---------------------------------------------------------------------------

async def process_task_bus(
    task_info: Dict[str, Any],
    sandbox_client: SandboxHTTPClient,
    semaphore: asyncio.Semaphore,
    max_rounds: int,
    max_steps: Optional[int],
    step_timeout: Optional[int],
    include_skills: bool,
    result_saver: BenchmarkResultSaver,
    total_tasks: int,
    completed_count_ref: list,
    completed_lock: asyncio.Lock,
    tracer: Optional["SkillsBenchTracer"] = None,
) -> Dict[str, Any]:
    """Submit one SkillsBench task to the AgentOS bus.

    The bus agents use bash_tool which runs locally. To bridge this with the
    sandbox, we inject the task instruction and tell the agent that all shell
    commands should be prefixed with the sandbox exec wrapper. The agent's
    bash_tool calls are intercepted by providing a wrapper script.

    Alternatively (simpler approach used here): we give the bus a task
    description that includes the instruction + sandbox interaction guidance,
    and let the planner/sub-agents reason about the task. The final result
    from the bus is then used to determine if the agent solved the task.
    For actual command execution we do a post-hoc rollout using the bus result.
    """
    from src.interaction import bus
    from src.task import Task
    from src.session import SessionContext

    task_id = task_info["task_id"]
    dataset = task_info["dataset"]
    meta = task_info["meta"]
    instruction = task_info["instruction"]

    effective_max_steps = max_steps or meta["default_max_steps"]
    effective_step_timeout = step_timeout or meta.get("agent_timeout_sec", 60)

    result = {
        "task_id": task_id,
        "dataset": dataset,
        "difficulty": meta.get("difficulty", ""),
        "reward": 0.0,
        "done": False,
        "steps": 0,
        "processing_time": 0.0,
        "commands": [],
    }

    # Begin trajectory tracing
    if tracer:
        await tracer.begin_task(task_id, dataset, meta.get("difficulty", ""), meta)

    async with semaphore:
        start_time = time.time()
        env_id = None
        task_error: Optional[str] = None
        try:
            logger.info(f"| {'='*50}")
            logger.info(f"| Processing Task (bus): {task_id}")
            logger.info(f"| {'='*50}")

            # 1. Create sandbox environment
            create_resp = await asyncio.to_thread(
                sandbox_client.create_env,
                {"dataset": dataset, "task_id": task_id},
            )
            env_id = create_resp["env_id"]
            logger.info(f"| Created env: {env_id}")

            # 2. Get instruction
            instr_resp = await asyncio.to_thread(sandbox_client.get_instruction, env_id)
            sandbox_instruction = instr_resp.get("instruction", instruction)

            # 3. Build task content for the bus
            skills_section = ""
            if include_skills and meta.get("skills_metadata"):
                skills_section = _format_skills_section(meta["skills_metadata"])

            bus_task_content = (
                f"You are solving a SkillsBench sandbox task. The task runs inside a Docker container "
                f"accessible via a sandbox server at env_id={env_id}.\n\n"
                f"## Task Instruction\n\n{sandbox_instruction}\n\n"
                f"{skills_section}\n\n"
                f"## Execution Environment\n\n"
                f"All commands must be executed inside the sandbox container. "
                f"To run a command, use the bash_tool with a curl command like:\n"
                f'curl -s -X POST http://127.0.0.1:8080/envs/{env_id}/step '
                f'-H "Content-Type: application/json" '
                f'-d \'{{"command": "<YOUR_COMMAND>", "timeout_sec": {effective_step_timeout}}}\'\n\n'
                f"The response JSON contains: exit_code, stdout, stderr.\n\n"
                f"When done, evaluate your work with:\n"
                f'curl -s -X POST http://127.0.0.1:8080/envs/{env_id}/evaluate '
                f'-H "Content-Type: application/json" -d \'{{}}\'\n\n'
                f"The evaluation response contains: reward (0.0-1.0), done (true if solved).\n\n"
                f"You have at most {effective_max_steps} command steps. "
                f"Work step by step, verify your progress, and aim for reward=1.0."
            )

            ctx = SessionContext(id=task_id)
            bus_task = Task(content=bus_task_content, session_id=ctx.id)

            try:
                response = await asyncio.wait_for(
                    bus.submit(bus_task, ctx=ctx, max_rounds=max_rounds),
                    timeout=3600.0,
                )
                bus_result = str(response.payload.get("result") or response.payload.get("error") or "")
                logger.info(f"| [{task_id}] Bus result: {bus_result[:200]}")
            except asyncio.TimeoutError:
                logger.error(f"| [{task_id}] Bus timeout (3600s)")
                bus_result = ""

            # Trace per-round planner records from the internal bus tracer
            if tracer:
                from src.config import config as _cfg
                _tracer_path = os.path.join(_cfg.workdir, "tracer", f"{task_id}.json")
                _round_steps_recorded = False
                if os.path.exists(_tracer_path):
                    try:
                        with open(_tracer_path, "r", encoding="utf-8") as _tf:
                            _tdata = json.load(_tf)
                        _records = []
                        for _sess_records in _tdata.get("sessions", {}).values():
                            _records.extend(_sess_records)
                        _records.sort(key=lambda r: r.get("id", 0))

                        for _step_idx, _rec in enumerate(_records, start=1):
                            _obs = _rec.get("observation") or {}
                            _act = _rec.get("action") or {}
                            _round_num = _obs.get("round", _step_idx)
                            _plan_update = _obs.get("plan_update", "")
                            _status = _obs.get("status", "")

                            # Build llm_output from action details
                            if "agents" in _act:
                                _parts = []
                                for _a in _act["agents"]:
                                    _tag = "OK" if _a.get("success") else "FAIL"
                                    _parts.append(
                                        f"[{_tag}] {_a.get('agent_name', '?')}: "
                                        f"{(_a.get('result') or _a.get('error') or '')[:500]}"
                                    )
                                _llm_out = "\n".join(_parts)
                            elif "final_result" in _act:
                                _llm_out = str(_act["final_result"])
                            elif "error" in _act:
                                _llm_out = f"ERROR: {_act['error']}"
                            else:
                                _llm_out = json.dumps(_act, ensure_ascii=False)[:500]

                            # Build command summary
                            if "agents" in _act:
                                _cmd = "dispatch: " + ", ".join(
                                    _a.get("agent_name", "?") for _a in _act["agents"]
                                )
                            else:
                                _cmd = f"(planner round {_round_num})"

                            await tracer.record_step(
                                task_id, _step_idx,
                                prompt=f"[Round {_round_num}] Plan: {_plan_update}" if _plan_update else f"[Round {_round_num}]",
                                llm_output=_llm_out,
                                command=_cmd,
                                is_done=(_status == "done"),
                                note=f"bus round {_round_num}, status={_status}",
                            )
                        if _records:
                            _round_steps_recorded = True
                    except Exception as _te:
                        logger.warning(f"| [{task_id}] Failed to read bus tracer: {_te}")

                # Fallback: record a single summary step if per-round parsing failed
                if not _round_steps_recorded:
                    await tracer.record_step(
                        task_id, 1,
                        prompt=bus_task_content,
                        llm_output=bus_result if bus_result else "",
                        command="(bus mode)",
                        note="bus pipeline result (no per-round data)",
                    )

            # 4. Evaluate (in case the agent already evaluated, we do it again to be sure)
            logger.info(f"| [{task_id}] Evaluating...")
            eval_resp = await asyncio.to_thread(
                sandbox_client.evaluate,
                env_id,
                {"timeout_sec": meta.get("verifier_timeout_sec", 600)},
            )
            raw_reward = eval_resp.get("reward")
            reward = float(raw_reward) if raw_reward is not None else 0.0
            done = bool(eval_resp.get("done", False))
            result["reward"] = reward
            result["done"] = done

            tag = "✅ Solved" if done else f"❌ reward={reward}"
            logger.info(f"| {tag} [{task_id}]")

        except Exception as exc:
            logger.error(f"| [{task_id}] Error: {exc}")
            task_error = str(exc)

        finally:
            result["processing_time"] = time.time() - start_time

            if env_id:
                try:
                    await asyncio.to_thread(sandbox_client.delete_env, env_id, False)
                except Exception:
                    pass

            await result_saver.add_task_result(result)

            # Finalize trajectory and flush to disk
            if tracer:
                await tracer.end_task(
                    task_id,
                    reward=result["reward"],
                    done=result["done"],
                    total_steps=result["steps"],
                    processing_time=result["processing_time"],
                    error=task_error,
                )

            async with completed_lock:
                completed_count_ref[0] += 1
                done_count = completed_count_ref[0]
                pct = done_count / total_tasks * 100
                logger.info(f"| Progress: {done_count}/{total_tasks} ({pct:.1f}%)")

    return result


# ---------------------------------------------------------------------------
# Task filtering (resume support)
# ---------------------------------------------------------------------------

def apply_filter(
    all_tasks: list,
    prev_results: list,
    prev_by_id: dict,
    completed_ids: set,
    resume_file: Optional[str],
    filter_mode: Optional[str],
    result_saver: BenchmarkResultSaver,
) -> list:
    if not resume_file:
        return all_tasks

    if filter_mode == "wrong":
        rerun_ids = {tid for tid, r in prev_by_id.items() if r.get("reward", 0) != 1.0}
        tasks_to_run = [t for t in all_tasks if t["task_id"] not in prev_by_id or t["task_id"] in rerun_ids]
        keep_ids = set(prev_by_id.keys()) - rerun_ids
        logger.info(f"| filter=wrong: re-running {len(tasks_to_run)}, skipping {len(keep_ids)} solved")
        result_saver.preload_results([r for r in prev_results if r["task_id"] in keep_ids])

    elif filter_mode == "null":
        rerun_ids = {tid for tid, r in prev_by_id.items() if r.get("reward") is None or r.get("steps", 0) == 0}
        tasks_to_run = [t for t in all_tasks if t["task_id"] not in prev_by_id or t["task_id"] in rerun_ids]
        keep_ids = set(prev_by_id.keys()) - rerun_ids
        logger.info(f"| filter=null: re-running {len(tasks_to_run)}, skipping {len(keep_ids)} with results")
        result_saver.preload_results([r for r in prev_results if r["task_id"] in keep_ids])

    else:
        tasks_to_run = [t for t in all_tasks if t["task_id"] not in completed_ids]
        logger.info(f"| resume: running {len(tasks_to_run)}, skipping {len(all_tasks) - len(tasks_to_run)} completed")
        result_saver.preload_results(prev_results)

    return tasks_to_run


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

async def run(
    tasks: List[Dict[str, Any]],
    sandbox_client: SandboxHTTPClient,
    max_concurrency: int,
    max_steps: Optional[int],
    max_rounds: int,
    step_timeout: Optional[int],
    include_skills: bool,
    result_saver: BenchmarkResultSaver,
    use_bus: bool,
    model_name: str,
    resume_file: Optional[str] = None,
    filter_mode: Optional[str] = None,
    tracer: Optional[SkillsBenchTracer] = None,
) -> None:
    # Load previous results for resume / filter
    completed_ids: set = set()
    prev_results = []
    prev_by_id: dict = {}
    if resume_file:
        if not os.path.exists(resume_file):
            logger.error(f"| Resume file not found: {resume_file}")
            return
        with open(resume_file, encoding="utf-8") as f:
            prev_data = json.load(f)
        prev_results = prev_data.get("results", [])
        for r in prev_results:
            completed_ids.add(r["task_id"])
            prev_by_id[r["task_id"]] = r
        logger.info(f"| Resume: loaded {len(prev_results)} previous results")

    tasks_to_run = apply_filter(tasks, prev_results, prev_by_id, completed_ids, resume_file, filter_mode, result_saver)

    total_tasks = len(tasks)
    result_saver.update_total_tasks(total_tasks)
    mode = "bus" if use_bus else "direct LLM"
    logger.info(f"| Collected {total_tasks} tasks ({len(tasks_to_run)} to run). Mode={mode}, concurrency={max_concurrency}")

    semaphore = asyncio.Semaphore(max_concurrency)
    preloaded = total_tasks - len(tasks_to_run)
    completed_count_ref = [preloaded]
    completed_lock = asyncio.Lock()

    if use_bus:
        coros = [
            process_task_bus(
                t, sandbox_client, semaphore, max_rounds, max_steps, step_timeout,
                include_skills, result_saver, total_tasks, completed_count_ref, completed_lock,
                tracer=tracer,
            )
            for t in tasks_to_run
        ]
    else:
        coros = [
            process_task_direct(
                t, sandbox_client, semaphore, model_name, max_steps, step_timeout,
                include_skills, result_saver, total_tasks, completed_count_ref, completed_lock,
                tracer=tracer,
            )
            for t in tasks_to_run
        ]

    await asyncio.gather(*coros, return_exceptions=True)

    # Write trajectory summary
    if tracer:
        summary_path = await tracer.write_summary(model_name=model_name)
        logger.info(f"| Trajectories saved to: {tracer.out_dir}")
        logger.info(f"| Trajectory summary: {summary_path}")

    # Final stats
    results = result_saver.results_data["results"]
    solved = sum(1 for r in results if r.get("reward", 0) == 1.0)
    total = len(results)
    avg_time = sum(r.get("processing_time", 0) for r in results) / total if total else 0
    avg_steps = sum(r.get("steps", 0) for r in results) / total if total else 0

    logger.info(f"| {'='*50}")
    logger.info("| Final Statistics")
    logger.info(f"| {'='*50}")
    logger.info(f"| Total: {total} | Solved: {solved} | Accuracy: {solved/total:.2%}" if total else "| No results")
    logger.info(f"| Avg time: {avg_time:.1f}s | Avg steps: {avg_steps:.1f}")
    logger.info(f"| Results saved to: {result_saver.get_file_path()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    args = parse_args()

    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
    logger.info(f"| Config: {config.pretty_text}")

    model_name = args.model_name or config.model_name

    # Check sandbox server health
    if not _is_server_healthy(args.server_url):
        logger.error(
            f"| SkillsBench sandbox server is not reachable at {args.server_url}. "
            f"Start it first: cd {SKILLSBENCH_PROJECT_ROOT} && python3 start_sandbox_server.py --config ./sandbox_config.yaml"
        )
        return

    sandbox_client = SandboxHTTPClient(args.server_url)
    logger.info(f"| Sandbox server healthy at {args.server_url}")

    logger.info("| Initializing version manager...")
    await version_manager.initialize()

    logger.info("| Initializing model manager...")
    await model_manager.initialize()

    if args.use_bus:
        from src.prompt import prompt_manager
        from src.memory import memory_manager
        from src.tool import tool_manager
        from src.skill import skill_manager
        from src.agent import agent_manager
        from src.interaction import bus

        logger.info("| Initializing prompt manager...")
        await prompt_manager.initialize()

        logger.info("| Initializing memory manager...")
        await memory_manager.initialize(memory_names=config.memory_names)

        logger.info("| Initializing tools...")
        await tool_manager.initialize(tool_names=config.tool_names)

        logger.info("| Initializing skills...")
        skill_names = getattr(config, "skill_names", None)
        await skill_manager.initialize(skill_names=skill_names)

        logger.info("| Initializing agents...")
        await agent_manager.initialize(agent_names=config.agent_names)
        logger.info(f"| Agents ready: {await agent_manager.list()}")

        logger.info("| Initializing AgentBus...")
        await bus.initialize()
        logger.info(f"| Bus agents: {await bus.list()}")

    # Discover tasks
    task_ids = [args.task_id] if args.task_id else None
    all_tasks = discover_tasks(args.dataset, task_ids=task_ids)
    if not all_tasks:
        logger.error(f"| No tasks found for dataset={args.dataset}")
        return

    # Apply slice
    if args.start is not None or args.end is not None:
        all_tasks = all_tasks[args.start:args.end]

    logger.info(f"| Discovered {len(all_tasks)} tasks from dataset={args.dataset}")

    result_saver = BenchmarkResultSaver("skillsbench", args.max_concurrency, len(all_tasks), model_name, run_name=args.run_name)
    logger.info(f"| Results will be saved to: {result_saver.get_file_path()}")

    # Create trajectory tracer (enabled by default)
    tracer: Optional[SkillsBenchTracer] = None
    if not args.no_trace:
        run_id = Path(result_saver.get_file_path()).stem  # e.g. benchmark_skillsbench_2026-...
        traj_base = os.path.join(config.workdir, "results", "skillsbench", "trajectories")
        tracer = SkillsBenchTracer(run_id=run_id, base_dir=traj_base)
        logger.info(f"| Trajectory tracing enabled -> {tracer.out_dir}")

    if args.filter and not args.resume:
        logger.warning("| --filter has no effect without --resume, ignoring")

    # Resolve resume file
    resume_file = None
    if args.resume:
        results_dir = os.path.join(config.workdir, "results", "skillsbench")
        candidates = sorted(Path(results_dir).glob("benchmark_skillsbench_*.json")) if os.path.isdir(results_dir) else []
        if not candidates:
            logger.error(f"| --resume: no previous results found in {results_dir}")
            return
        resume_file = str(candidates[-1])
        logger.info(f"| --resume: using {resume_file}")

    await run(
        tasks=all_tasks,
        sandbox_client=sandbox_client,
        max_concurrency=args.max_concurrency,
        max_steps=args.max_steps,
        max_rounds=args.max_rounds,
        step_timeout=args.step_timeout,
        include_skills=not args.disable_skill_injection,
        result_saver=result_saver,
        use_bus=args.use_bus,
        model_name=model_name,
        resume_file=resume_file,
        filter_mode=args.filter if args.resume else None,
        tracer=tracer,
    )

    if args.use_bus:
        from src.interaction import bus
        await bus.shutdown()

    logger.info("| Done.")


if __name__ == "__main__":
    asyncio.run(main())
