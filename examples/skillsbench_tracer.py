"""Trajectory tracer and per-trial results tracker for SkillsBench.

Captures the full interaction trajectory for every task trial:
  - LLM prompts and raw responses
  - Extracted commands and sandbox execution results
  - Evaluation outcomes and timing

Each task's trajectory is flushed to its own JSON file on completion,
and a run-level summary is written at the end.

Directory layout (under workdir/results/skillsbench/trajectories/):
    <run_id>/
        <task_id>.json      # per-task trajectory
        summary.json        # aggregate run summary

Usage:
    tracer = SkillsBenchTracer(run_id="benchmark_skillsbench_2026-...", base_dir="...")
    tracer.begin_task(task_id, dataset, difficulty, meta)

    # inside the step loop:
    tracer.record_step(task_id, step_idx, prompt_messages, llm_output, command,
                       sandbox_response, ...)

    tracer.end_task(task_id, reward, done, steps, processing_time)
    tracer.write_summary()
"""

import json
import os
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional


class StepRecord:
    """One step in a task trajectory."""

    __slots__ = (
        "step", "timestamp",
        "prompt", "llm_output", "command",
        "exit_code", "stdout", "stderr",
        "is_done", "note",
    )

    def __init__(
        self,
        step: int,
        prompt: Any = None,
        llm_output: str = "",
        command: str = "",
        exit_code: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
        is_done: bool = False,
        note: str = "",
    ):
        self.step = step
        self.timestamp = datetime.now().isoformat()
        self.prompt = prompt
        self.llm_output = llm_output
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.is_done = is_done
        self.note = note

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "step": self.step,
            "timestamp": self.timestamp,
            "llm_output": self.llm_output,
            "command": self.command,
        }
        if self.prompt:
            d["prompt"] = self.prompt
        if self.exit_code is not None:
            d["exit_code"] = self.exit_code
            d["stdout"] = self.stdout
            d["stderr"] = self.stderr
        if self.is_done:
            d["is_done"] = True
        if self.note:
            d["note"] = self.note
        return d


class TaskTrajectory:
    """Full trajectory for a single task trial."""

    def __init__(
        self,
        task_id: str,
        dataset: str = "",
        difficulty: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id
        self.dataset = dataset
        self.difficulty = difficulty
        self.meta = meta or {}
        self.start_time = datetime.now().isoformat()
        self.end_time: Optional[str] = None
        self.steps: List[StepRecord] = []

        # Final evaluation (populated by end_task)
        self.reward: float = 0.0
        self.done: bool = False
        self.total_steps: int = 0
        self.processing_time: float = 0.0
        self.error: Optional[str] = None

    def add_step(self, record: StepRecord) -> None:
        self.steps.append(record)

    def finalize(
        self,
        reward: float,
        done: bool,
        total_steps: int,
        processing_time: float,
        error: Optional[str] = None,
    ) -> None:
        self.end_time = datetime.now().isoformat()
        self.reward = reward
        self.done = done
        self.total_steps = total_steps
        self.processing_time = processing_time
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "task_id": self.task_id,
            "dataset": self.dataset,
            "difficulty": self.difficulty,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "reward": self.reward,
            "done": self.done,
            "total_steps": self.total_steps,
            "processing_time": self.processing_time,
            "trajectory": [s.to_dict() for s in self.steps],
        }
        if self.error:
            d["error"] = self.error
        if self.meta:
            # Include select meta fields (avoid huge blobs)
            d["meta"] = {
                k: v for k, v in self.meta.items()
                if k in ("default_max_steps", "agent_timeout_sec", "verifier_timeout_sec")
            }
        return d


class SkillsBenchTracer:
    """Manages trajectory collection for an entire benchmark run.

    Thread/coroutine-safe: each task operates on its own trajectory object
    under an asyncio lock.
    """

    def __init__(self, run_id: str, base_dir: str):
        """
        Args:
            run_id:   Identifier for this run (e.g. the result filename stem).
            base_dir: Parent directory for trajectory output
                      (e.g. <workdir>/results/skillsbench/trajectories).
        """
        self.run_id = run_id
        self.out_dir = os.path.join(base_dir, run_id)
        os.makedirs(self.out_dir, exist_ok=True)

        self._trajectories: Dict[str, TaskTrajectory] = {}
        self._lock = asyncio.Lock()

    # ---- lifecycle per task ------------------------------------------------

    async def begin_task(
        self,
        task_id: str,
        dataset: str = "",
        difficulty: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        async with self._lock:
            self._trajectories[task_id] = TaskTrajectory(
                task_id=task_id,
                dataset=dataset,
                difficulty=difficulty,
                meta=meta,
            )

    async def record_step(
        self,
        task_id: str,
        step: int,
        prompt: Any = None,
        llm_output: str = "",
        command: str = "",
        exit_code: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
        is_done: bool = False,
        note: str = "",
    ) -> None:
        """Record a single interaction step for *task_id*."""
        record = StepRecord(
            step=step,
            prompt=prompt,
            llm_output=llm_output,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            is_done=is_done,
            note=note,
        )
        async with self._lock:
            traj = self._trajectories.get(task_id)
            if traj is not None:
                traj.add_step(record)

    async def end_task(
        self,
        task_id: str,
        reward: float = 0.0,
        done: bool = False,
        total_steps: int = 0,
        processing_time: float = 0.0,
        error: Optional[str] = None,
    ) -> None:
        """Finalize a task trajectory and flush it to disk."""
        async with self._lock:
            traj = self._trajectories.get(task_id)
            if traj is None:
                return
            traj.finalize(
                reward=reward,
                done=done,
                total_steps=total_steps,
                processing_time=processing_time,
                error=error,
            )
        # Write to individual JSON file (outside the lock to avoid blocking)
        self._flush_task(task_id, traj)

    # ---- persistence -------------------------------------------------------

    def _flush_task(self, task_id: str, traj: TaskTrajectory) -> None:
        """Write a single task trajectory to disk."""
        # Sanitize task_id for filename (replace path separators)
        safe_name = task_id.replace("/", "_").replace("\\", "_")
        path = os.path.join(self.out_dir, f"{safe_name}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(traj.to_dict(), f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass  # best-effort; the main result saver is the authority

    async def write_summary(self, model_name: str = "", extra: Optional[Dict[str, Any]] = None) -> str:
        """Write a run-level summary.json and return its path."""
        async with self._lock:
            trajs = list(self._trajectories.values())

        total = len(trajs)
        solved = sum(1 for t in trajs if t.done)
        total_reward = sum(t.reward for t in trajs)
        avg_steps = sum(t.total_steps for t in trajs) / total if total else 0
        avg_time = sum(t.processing_time for t in trajs) / total if total else 0

        summary: Dict[str, Any] = {
            "run_id": self.run_id,
            "model": model_name,
            "timestamp": datetime.now().isoformat() + "Z",
            "total_tasks": total,
            "solved": solved,
            "accuracy": solved / total if total else 0.0,
            "total_reward": total_reward,
            "avg_steps": round(avg_steps, 2),
            "avg_processing_time": round(avg_time, 2),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "dataset": t.dataset,
                    "difficulty": t.difficulty,
                    "reward": t.reward,
                    "done": t.done,
                    "total_steps": t.total_steps,
                    "processing_time": round(t.processing_time, 2),
                    "error": t.error,
                }
                for t in sorted(trajs, key=lambda x: x.task_id)
            ],
        }
        if extra:
            summary["extra"] = extra

        path = os.path.join(self.out_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        return path
