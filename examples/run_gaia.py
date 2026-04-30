"""Run the GAIA benchmark through the AgentBus."""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mmengine import DictAction

load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.benchmark import benchmark_manager
from src.benchmark.types import Task as BenchmarkTask
from src.config import config
from src.logger import logger
from src.model import model_manager
from src.version import version_manager

GAIA_TASK_SUFFIX = (
    "\n\nReturn only the final answer, concise and with no explanation. "
    "If the problem asks for a specific format, follow it exactly. "
    "If a file is attached, use it as part of solving the task."
)

ANSWER_PATTERNS = [
    re.compile(r"final\s+answer\s*:\s*(.+)", re.IGNORECASE | re.DOTALL),
    re.compile(r"^answer\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE | re.DOTALL),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run GAIA benchmark through the AgentBus")
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs", "v3_bus.py"),
        help="config file path",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="GAIA split to run",
    )
    parser.add_argument(
        "--level",
        choices=["all", "1", "2", "3"],
        default="all",
        help="GAIA difficulty level to run",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="optional override for local GAIA dataset root",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="maximum concurrent GAIA tasks",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=20,
        help="maximum planner rounds per task",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="start index of the GAIA subset (inclusive)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="end index of the GAIA subset (exclusive)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from the latest GAIA results file",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["wrong", "null", "none"],
        default="none",
        help=(
            "requires --resume. Filter mode: "
            "'wrong' re-runs tasks where correct=False; "
            "'null' re-runs tasks with empty or failed outputs; "
            "'none' skips already-completed tasks and runs the remaining ones."
        ),
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override config settings in xxx=yyy format",
    )
    return parser.parse_args()


def extract_final_answer(raw_result: str) -> str:
    text = str(raw_result or "").strip()
    if not text:
        return ""

    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                candidate = candidate.splitlines()[0].strip()
                return candidate.strip("` ").strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[-1].strip("` ").strip()


def is_null_like_result(record: dict) -> bool:
    predicted = str(record.get("predicted_answer", "") or "").strip().lower()
    raw_result = str(record.get("raw_result", "") or "").strip().lower()
    bus_error = str(record.get("bus_error", "") or "").strip()
    if not predicted:
        return True
    if bus_error:
        return True
    failure_markers = [
        "planner did not finish",
        "unable to determine",
        "timeout",
        "timed out",
        "task failed",
        "error:",
    ]
    return any(marker in predicted or marker in raw_result for marker in failure_markers)


class BenchmarkResultSaver:
    """Save GAIA benchmark results to JSON with real-time updates."""

    def __init__(self, benchmark_name: str, concurrency: int, total_tasks: int, model_name: str, split: str, level: str):
        self.benchmark_name = benchmark_name
        self.start_time = datetime.now()

        results_dir = os.path.join(config.workdir, "results", benchmark_name)
        os.makedirs(results_dir, exist_ok=True)

        timestamp = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        self.filepath = os.path.join(results_dir, f"benchmark_{benchmark_name}_{timestamp}.json")
        self.file_lock = asyncio.Lock()

        self.results_data = {
            "experiment_meta": {
                "timestamp": self.start_time.isoformat() + "Z",
                "benchmark": benchmark_name,
                "concurrency": concurrency,
                "total_tasks": total_tasks,
                "model": model_name,
                "split": split,
                "level": level,
            },
            "results": [],
            "summary": {
                "completed_tasks": 0,
                "scored_tasks": 0,
                "correct_answers": 0,
                "accuracy": None,
                "last_updated": self.start_time.isoformat() + "Z",
            },
        }

    def _recompute_summary(self) -> None:
        results = self.results_data["results"]
        scored = [r for r in results if r.get("correct") is not None]
        correct = sum(1 for r in scored if r.get("correct") is True)
        self.results_data["summary"].update({
            "completed_tasks": len(results),
            "scored_tasks": len(scored),
            "correct_answers": correct,
            "accuracy": (correct / len(scored)) if scored else None,
            "last_updated": datetime.now().isoformat() + "Z",
        })

    def update_total_tasks(self, total_tasks: int) -> None:
        self.results_data["experiment_meta"]["total_tasks"] = total_tasks

    def preload_results(self, previous_results: list) -> None:
        self.results_data["results"] = sorted(previous_results, key=lambda r: r.get("task_id", ""))
        self._recompute_summary()
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.results_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            logger.info(f"| Preloaded {len(previous_results)} results saved to: {self.filepath}")
        except Exception as exc:
            logger.error(f"Failed to save preloaded results: {exc}")

    async def add_task_result(self, task: BenchmarkTask, processing_time: float = 0.0) -> None:
        async with self.file_lock:
            extra = task.extra or {}
            correct = None if task.score is None else task.score == 1.0
            self.results_data["results"].append({
                "task_id": task.task_id,
                "task_input": task.input,
                "predicted_answer": str(task.result) if task.result is not None else "",
                "raw_result": str(extra.get("raw_result", "") or ""),
                "ground_truth": str(task.ground_truth) if task.ground_truth is not None else "",
                "correct": correct,
                "level": str(extra.get("level", "")),
                "attachment_path": str(extra.get("attachment_path", "") or ""),
                "processing_time": processing_time,
                "bus_error": str(extra.get("bus_error", "") or ""),
            })
            self._recompute_summary()
            try:
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump(self.results_data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as exc:
                logger.error(f"Failed to save results: {exc}")

    def get_file_path(self) -> str:
        return str(self.filepath)


def apply_filter(
    all_tasks: list,
    prev_results: list,
    prev_by_id: dict,
    completed_ids: set,
    resume_file: Optional[str],
    filter_mode: Optional[str],
    result_saver: BenchmarkResultSaver,
    split: str,
) -> list:
    if not resume_file:
        return all_tasks

    effective_filter = filter_mode
    if split == "test" and effective_filter == "wrong":
        logger.warning("| split=test has no local correctness labels in saved runs; downgrading filter=wrong to filter=null")
        effective_filter = "null"

    if effective_filter == "wrong":
        rerun_ids = {task_id for task_id, record in prev_by_id.items() if record.get("correct") is False}
        tasks_to_run = [task for task in all_tasks if task.task_id not in prev_by_id or task.task_id in rerun_ids]
        keep_ids = set(prev_by_id.keys()) - rerun_ids
        logger.info(f"| filter=wrong: re-running {len(tasks_to_run)}, skipping {len(keep_ids)} correct")
        result_saver.preload_results([record for record in prev_results if record["task_id"] in keep_ids])
        return tasks_to_run

    if effective_filter == "null":
        rerun_ids = {task_id for task_id, record in prev_by_id.items() if is_null_like_result(record)}
        tasks_to_run = [task for task in all_tasks if task.task_id not in prev_by_id or task.task_id in rerun_ids]
        keep_ids = set(prev_by_id.keys()) - rerun_ids
        logger.info(f"| filter=null: re-running {len(tasks_to_run)}, skipping {len(keep_ids)} with usable answers")
        result_saver.preload_results([record for record in prev_results if record["task_id"] in keep_ids])
        return tasks_to_run

    tasks_to_run = [task for task in all_tasks if task.task_id not in completed_ids]
    logger.info(f"| resume: running {len(tasks_to_run)}, skipping {len(all_tasks) - len(tasks_to_run)} completed")
    result_saver.preload_results(prev_results)
    return tasks_to_run


async def process_task_bus(
    bench_task: BenchmarkTask,
    semaphore: asyncio.Semaphore,
    max_rounds: int,
    result_saver: BenchmarkResultSaver,
    total_tasks: int,
    completed_count_ref: list,
    completed_lock: asyncio.Lock,
) -> BenchmarkTask:
    from src.interaction import bus
    from src.session import SessionContext
    from src.task import Task

    task_id = bench_task.task_id

    async with semaphore:
        start_time = time.time()
        bus_error = ""
        raw_result = ""
        try:
            logger.info(f"| {'=' * 50}")
            logger.info(f"| Processing Task (bus): {task_id}")
            logger.info(f"| {'=' * 50}")

            content = bench_task.input.rstrip() + GAIA_TASK_SUFFIX
            files = []
            attachment_path = (bench_task.extra or {}).get("attachment_path", "")
            if attachment_path:
                files = [attachment_path]
                content = f"{content}\n\n[Attached file: {os.path.basename(attachment_path)}]"

            ctx = SessionContext(id=task_id)
            bus_task = Task(content=content, session_id=ctx.id, files=files)

            try:
                response = await asyncio.wait_for(
                    bus.submit(bus_task, ctx=ctx, max_rounds=max_rounds),
                    timeout=3600.0,
                )
                raw_result = str(response.payload.get("result") or "")
                bus_error = str(response.payload.get("error") or "")
            except asyncio.TimeoutError:
                bus_error = "Bus timeout (3600s)"
                logger.error(f"[Task {task_id}] {bus_error}")

            predicted_answer = extract_final_answer(raw_result) if raw_result else ""
            bench_task.result = predicted_answer
            bench_task.reasoning = ""
            bench_task.extra = {**(bench_task.extra or {}), "raw_result": raw_result, "bus_error": bus_error}

            try:
                bench_task = await asyncio.wait_for(
                    benchmark_manager.eval("gaia", bench_task),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.error(f"[Task {task_id}] Eval timeout")
            except Exception as exc:
                logger.error(f"[Task {task_id}] Eval error: {exc}")

            bench_task.time = time.time() - start_time
            if bench_task.score is None:
                tag = "ℹ️ Unscored"
            else:
                tag = "✅ Correct" if bench_task.score == 1.0 else "❌ Wrong"
            logger.info(f"| {tag} [{task_id}] | score={bench_task.score} | time={bench_task.time:.1f}s")

        except Exception as exc:
            logger.error(f"[Task {task_id}] Unexpected error: {exc}")
            bench_task.time = time.time() - start_time
            bench_task.extra = {**(bench_task.extra or {}), "bus_error": str(exc), "raw_result": raw_result}

        finally:
            await result_saver.add_task_result(bench_task, bench_task.time or 0.0)
            async with completed_lock:
                completed_count_ref[0] += 1
                done = completed_count_ref[0]
                pct = done / total_tasks * 100 if total_tasks else 100.0
                logger.info(f"| Progress: {done}/{total_tasks} ({pct:.1f}%)")

    return bench_task


async def run(
    split: str,
    level: str,
    max_concurrency: int,
    max_rounds: int,
    result_saver: BenchmarkResultSaver,
    resume_file: Optional[str] = None,
    filter_mode: Optional[str] = None,
) -> None:
    save_dir = os.path.join(config.workdir, "benchmark", "gaia")
    os.makedirs(save_dir, exist_ok=True)

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
        for record in prev_results:
            completed_ids.add(record["task_id"])
            prev_by_id[record["task_id"]] = record
        logger.info(f"| Resume: loaded {len(prev_results)} previous results")

    logger.info("| Resetting GAIA benchmark...")
    task = await benchmark_manager.reset("gaia")
    if not task:
        logger.warning("No GAIA tasks available.")
        return

    all_tasks = []
    while task is not None:
        all_tasks.append(task)
        task = await benchmark_manager.step("gaia")

    tasks_to_run = apply_filter(
        all_tasks,
        prev_results,
        prev_by_id,
        completed_ids,
        resume_file,
        filter_mode,
        result_saver,
        split,
    )

    total_tasks = len(all_tasks)
    result_saver.update_total_tasks(total_tasks)
    logger.info(
        f"| Collected {total_tasks} GAIA tasks ({len(tasks_to_run)} to run). "
        f"split={split}, level={level}, concurrency={max_concurrency}"
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    preloaded = total_tasks - len(tasks_to_run)
    completed_count_ref = [preloaded]
    completed_lock = asyncio.Lock()

    coroutines = [
        process_task_bus(
            task,
            semaphore,
            max_rounds,
            result_saver,
            total_tasks,
            completed_count_ref,
            completed_lock,
        )
        for task in tasks_to_run
    ]
    await asyncio.gather(*coroutines, return_exceptions=True)

    logger.info(f"| {'=' * 50}")
    logger.info("| Final Statistics")
    logger.info(f"| {'=' * 50}")
    stats = await benchmark_manager.stats("gaia")
    if stats:
        attempted = stats.extra.get("attempted", stats.correct + stats.wrong) if stats.extra else stats.correct + stats.wrong
        logger.info(f"| Attempted: {attempted}/{stats.total}")
        logger.info(f"| Scored tasks: {stats.extra.get('scored_tasks', 0) if stats.extra else 0}")
        logger.info(f"| Unscored tasks: {stats.extra.get('unscored_tasks', 0) if stats.extra else 0}")
        if stats.extra and stats.extra.get("scored_tasks", 0):
            logger.info(f"| Accuracy: {stats.accuracy:.2%}")
            logger.info(f"| Correct: {stats.correct} | Wrong: {stats.wrong}")
            logger.info(f"| Per-level accuracy: {stats.extra.get('per_level_accuracy', {})}")
        logger.info(f"| Avg time: {stats.average_time:.2f}s")

        stats_path = os.path.join(save_dir, "stats.json")
        stats_data = stats.model_dump()
        stats_data["tasks"] = result_saver.results_data["results"]
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=4, ensure_ascii=False)
        logger.info(f"| Stats saved to: {stats_path}")


async def main():
    args = parse_args()

    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
    logger.info(f"| Config: {config.pretty_text}")

    logger.info("| Initializing version manager...")
    await version_manager.initialize()

    logger.info("| Initializing model manager...")
    await model_manager.initialize()

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

    logger.info("| Initializing benchmark manager (GAIA)...")
    gaia_benchmark_config = config.gaia_benchmark
    gaia_benchmark_config.update({
        "split": args.split,
        "level": args.level,
        "start": args.start,
        "end": args.end,
    })
    if args.dataset_path:
        gaia_benchmark_config["path"] = args.dataset_path
    await benchmark_manager.initialize(benchmark_names=["gaia"])

    result_saver = BenchmarkResultSaver(
        "gaia",
        args.max_concurrency,
        0,
        getattr(config, "model_name", ""),
        args.split,
        args.level,
    )
    logger.info(f"| Results will be saved to: {result_saver.get_file_path()}")

    if args.filter != "none" and not args.resume:
        logger.warning("| --filter has no effect without --resume, ignoring")

    resume_file = None
    if args.resume:
        results_dir = os.path.join(config.workdir, "results", "gaia")
        candidates = sorted(Path(results_dir).glob("benchmark_gaia_*.json")) if os.path.isdir(results_dir) else []
        if not candidates:
            logger.error(f"| --resume: no previous results found in {results_dir}")
            return
        resume_file = str(candidates[-1])
        logger.info(f"| --resume: using {resume_file}")

    await run(
        split=args.split,
        level=args.level,
        max_concurrency=args.max_concurrency,
        max_rounds=args.max_rounds,
        result_saver=result_saver,
        resume_file=resume_file,
        filter_mode=(args.filter if args.filter != "none" else None) if args.resume else None,
    )

    await bus.shutdown()
    await benchmark_manager.cleanup()
    logger.info("| Done.")


if __name__ == "__main__":
    asyncio.run(main())
