"""Run HLE benchmark — either direct LLM inference or through the AgentBus.

Usage:
    # Direct LLM (default, fast baseline):
    python examples/run_hle.py --model-name openrouter/gemini-3-flash-preview

    # Bus mode (full agent pipeline):
    python examples/run_hle.py --use-bus
    python examples/run_hle.py --use-bus --config configs/bus.py --max-rounds 10

    # Resume from latest results file, skip already-completed tasks:
    python examples/run_hle.py --use-bus --resume

    # Filter-wrong: only re-run tasks answered incorrectly in the previous run:
    python examples/run_hle.py --use-bus --resume --filter-wrong
"""

import asyncio
import base64
import json
import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mmengine import DictAction

load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.model import model_manager
from src.version import version_manager
from src.benchmark import benchmark_manager
from src.benchmark.types import Task as BenchmarkTask
from src.message.types import SystemMessage, HumanMessage, ContentPartText, ContentPartImage, ImageURL
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run HLE benchmark")
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs", "bus.py"),
        help="config file path",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="only run evaluation, do not perform training",
    )
    parser.add_argument(
        "--eval-model-name",
        type=str,
        default="openai/o3-mini",
        help="model name to use for evaluation",
    )
    parser.add_argument(
        "--model-name", type=str, default="newapi/gemini-3.1-pro-preview",
        help="model to use (default: config.model_name)",
    )
    parser.add_argument(
        "--use-bus", 
        action="store_true", 
        default=True,
        help="use the AgentBus pipeline instead of direct LLM inference",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=8,
        help="maximum concurrent tasks (default: 8)",
    )
    parser.add_argument(
        "--max-rounds", type=int, default=20,
        help="maximum planner rounds per task, bus mode only (default: 20)",
    )
    parser.add_argument(
        "--start", type=int, default=None,
        help="start index of HLE dataset subset (inclusive)",
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="end index of HLE dataset subset (exclusive)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="resume from the latest results JSON in workdir/results/hle/, skipping already-completed tasks",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["wrong", "null", "none"],
        default="none",
        help=(
            "requires --resume. Filter mode: "
            "'wrong' re-runs tasks where correct=False; "
            "'null' re-runs tasks where predicted_answer is empty; "
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


class HLEAnswer(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning process.")
    final_result: str = Field(
        description=(
            "Concise final answer with no extra text. "
            "For multiple-choice questions, output only the letter (e.g. 'A'). "
            "For exact-match questions, output as few words or numbers as possible. "
            "No units unless required. No trailing punctuation."
        )
    )

# ---------------------------------------------------------------------------
# Result saver
# ---------------------------------------------------------------------------

class BenchmarkResultSaver:
    """Save benchmark results to JSON with real-time updates."""

    def __init__(self, benchmark_name: str, concurrency: int, total_tasks: int, model_name: str):
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
            },
            "results": [],
            "summary": {
                "completed_tasks": 0,
                "correct_answers": 0,
                "accuracy": 0.0,
                "last_updated": self.start_time.isoformat() + "Z",
            },
        }

    def update_total_tasks(self, total_tasks: int) -> None:
        self.results_data["experiment_meta"]["total_tasks"] = total_tasks

    def preload_results(self, previous_results: list) -> None:
        """Pre-populate results from a previous run (for resume / filter-wrong) and flush to disk."""
        self.results_data["results"] = sorted(previous_results, key=lambda r: r.get("task_id", ""))
        results = self.results_data["results"]
        correct = sum(1 for r in results if r.get("correct", False))
        self.results_data["summary"].update({
            "completed_tasks": len(results),
            "correct_answers": correct,
            "accuracy": correct / len(results) if results else 0.0,
            "last_updated": datetime.now().isoformat() + "Z",
        })
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.results_data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            logger.info(f"| Preloaded {len(results)} results saved to: {self.filepath}")
        except Exception as exc:
            logger.error(f"Failed to save preloaded results: {exc}")

    async def add_task_result(self, task: BenchmarkTask, processing_time: float = 0.0) -> None:
        async with self.file_lock:
            self.results_data["results"].append({
                "task_id": task.task_id,
                "task_input": task.input,
                "image": task.extra.get("image", "") if task.extra else "",
                "ground_truth": str(task.ground_truth) if task.ground_truth else "",
                "predicted_answer": str(task.result) if task.result else "",
                "correct": task.score == 1.0 if task.score is not None else False,
                "processing_time": processing_time,
                "answer_type": task.extra.get("answer_type", "exactMatch") if task.extra else "exactMatch",
            })
            results = self.results_data["results"]
            correct = sum(1 for r in results if r["correct"])
            self.results_data["summary"].update({
                "completed_tasks": len(results),
                "correct_answers": correct,
                "accuracy": correct / len(results) if results else 0.0,
                "last_updated": datetime.now().isoformat() + "Z",
            })
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
# Direct LLM inference (no bus)
# ---------------------------------------------------------------------------

async def process_task_direct(
    bench_task: BenchmarkTask,
    semaphore: asyncio.Semaphore,
    model_name: str,
    result_saver: BenchmarkResultSaver,
    total_tasks: int,
    completed_count_ref: list,
    completed_lock: asyncio.Lock,
) -> BenchmarkTask:
    """Call the LLM directly (no bus/agents) to answer one HLE task."""
    task_id = bench_task.task_id

    async with semaphore:
        start_time = time.time()
        try:
            logger.info(f"| {'='*50}")
            logger.info(f"| Processing Task (direct): {task_id}")
            logger.info(f"| {'='*50}")

            # Build message content — include image if present
            user_content = bench_task.input
            if bench_task.extra and bench_task.extra.get("image"):
                image_path = bench_task.extra["image"]
                media_type = bench_task.extra.get("image_media_type", "image/jpeg")
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                data_uri = f"data:{media_type};base64,{b64}"
                image_part = ContentPartImage(
                    image_url=ImageURL(url=data_uri, media_type=media_type)
                )
                user_content = [ContentPartText(text=bench_task.input), image_part]

            messages = []
            if bench_task.system_prompt:
                messages.append(SystemMessage(content=bench_task.system_prompt))
            messages.append(HumanMessage(content=user_content))

            response = await model_manager(model=model_name, messages=messages, response_format=HLEAnswer)
            answer: HLEAnswer = response.extra.parsed_model
            bench_task.result = answer.final_result
            bench_task.reasoning = answer.reasoning

            # Evaluate
            try:
                bench_task = await asyncio.wait_for(
                    benchmark_manager.eval("hle", bench_task),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.error(f"[Task {task_id}] Eval timeout")
            except Exception as exc:
                logger.error(f"[Task {task_id}] Eval error: {exc}")

            bench_task.time = time.time() - start_time
            correct = bench_task.score == 1.0 if bench_task.score is not None else False
            tag = "✅ Correct" if correct else "❌ Wrong"
            logger.info(f"| {tag} [{task_id}] | score={bench_task.score} | time={bench_task.time:.1f}s")

        except Exception as exc:
            logger.error(f"[Task {task_id}] Unexpected error: {exc}")
            bench_task.time = time.time() - start_time

        finally:
            await result_saver.add_task_result(bench_task, bench_task.time or 0.0)
            async with completed_lock:
                completed_count_ref[0] += 1
                done = completed_count_ref[0]
                pct = done / total_tasks * 100
                logger.info(f"| Progress: {done}/{total_tasks} ({pct:.1f}%)")

    return bench_task


# ---------------------------------------------------------------------------
# Bus-based inference
# ---------------------------------------------------------------------------

async def process_task_bus(
    bench_task: BenchmarkTask,
    semaphore: asyncio.Semaphore,
    max_rounds: int,
    result_saver: BenchmarkResultSaver,
    total_tasks: int,
    completed_count_ref: list,
    completed_lock: asyncio.Lock,
) -> BenchmarkTask:
    """Submit one HLE task to the bus and evaluate the result."""
    from src.interaction import bus
    from src.task import Task
    from src.session import SessionContext

    task_id = bench_task.task_id

    async with semaphore:
        start_time = time.time()
        try:
            logger.info(f"| {'='*50}")
            logger.info(f"| Processing Task (bus): {task_id}")
            logger.info(f"| {'='*50}")

            content = bench_task.input
            files = []
            if bench_task.extra and bench_task.extra.get("image"):
                image_path = bench_task.extra["image"]
                files = [image_path]
                content = f"{content}\n\n[Image attached: {os.path.basename(image_path)}]"

            ctx = SessionContext(id=task_id)
            bus_task = Task(content=content, session_id=ctx.id, files=files)

            try:
                response = await asyncio.wait_for(
                    bus.submit(bus_task, ctx=ctx, max_rounds=max_rounds),
                    timeout=3600.0,
                )
                result_text = str(response.payload.get("result") or response.payload.get("error") or "")
            except asyncio.TimeoutError:
                logger.error(f"[Task {task_id}] Bus timeout (3600s)")
                result_text = ""

            bench_task.result = result_text
            bench_task.reasoning = ""

            try:
                bench_task = await asyncio.wait_for(
                    benchmark_manager.eval("hle", bench_task),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.error(f"[Task {task_id}] Eval timeout")
            except Exception as exc:
                logger.error(f"[Task {task_id}] Eval error: {exc}")

            bench_task.time = time.time() - start_time
            correct = bench_task.score == 1.0 if bench_task.score is not None else False
            tag = "✅ Correct" if correct else "❌ Wrong"
            logger.info(f"| {tag} [{task_id}] | score={bench_task.score} | time={bench_task.time:.1f}s")

        except Exception as exc:
            logger.error(f"[Task {task_id}] Unexpected error: {exc}")
            bench_task.time = time.time() - start_time

        finally:
            await result_saver.add_task_result(bench_task, bench_task.time or 0.0)
            async with completed_lock:
                completed_count_ref[0] += 1
                done = completed_count_ref[0]
                pct = done / total_tasks * 100
                logger.info(f"| Progress: {done}/{total_tasks} ({pct:.1f}%)")

    return bench_task


# ---------------------------------------------------------------------------
# Task filtering
# ---------------------------------------------------------------------------

def apply_filter(
    all_tasks: list,
    prev_results: list,
    prev_by_id: dict,
    completed_ids: set,
    resume_file: Optional[str],
    filter_mode: Optional[str],
    result_saver: "BenchmarkResultSaver",
) -> list:
    """Return the subset of tasks to run and preload skipped results into result_saver."""
    if not resume_file:
        return all_tasks

    if filter_mode == "wrong":
        rerun_ids = set()
        for tid, r in prev_by_id.items():
            if not r.get("correct", False):
                rerun_ids.add(tid)
        tasks_to_run = [t for t in all_tasks if t.task_id not in prev_by_id or t.task_id in rerun_ids]
        keep_ids = set(prev_by_id.keys()) - rerun_ids
        logger.info(f"| filter=wrong: re-running {len(tasks_to_run)}, skipping {len(keep_ids)} correct")
        result_saver.preload_results([r for r in prev_results if r["task_id"] in keep_ids])

    elif filter_mode == "null":
        rerun_ids = set()
        for tid, r in prev_by_id.items():
            if not r.get("predicted_answer", ""):
                rerun_ids.add(tid)
            if "planner did not finish" in r.get("predicted_answer", "").lower():
                rerun_ids.add(tid)
            if "unable to determine" in r.get("predicted_answer", "").lower():
                rerun_ids.add(tid)
            if r.get("predicted_answer", "").lower() in ["0", 0, "none"] and r.get("correct", False) is False:
                rerun_ids.add(tid)
        tasks_to_run = [t for t in all_tasks if t.task_id not in prev_by_id or t.task_id in rerun_ids]
        keep_ids = set(prev_by_id.keys()) - rerun_ids
        logger.info(f"| filter=null: re-running {len(tasks_to_run)}, skipping {len(keep_ids)} with answers")
        result_saver.preload_results([r for r in prev_results if r["task_id"] in keep_ids])

    else:
        tasks_to_run = [t for t in all_tasks if t.task_id not in completed_ids]
        logger.info(f"| resume: running {len(tasks_to_run)}, skipping {len(all_tasks) - len(tasks_to_run)} completed")
        result_saver.preload_results(prev_results)

    return tasks_to_run


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

async def run(
    max_concurrency: int,
    max_rounds: int,
    result_saver: BenchmarkResultSaver,
    use_bus: bool,
    model_name: str,
    resume_file: Optional[str] = None,
    filter_mode: Optional[str] = None,  # "wrong" | "null" | None
) -> None:
    save_dir = os.path.join(config.workdir, "benchmark", "hle")
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load previous results for resume / filter
    # ------------------------------------------------------------------
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

    logger.info("| Resetting HLE benchmark...")
    task = await benchmark_manager.reset("hle")
    if not task:
        logger.warning("No HLE tasks available.")
        return

    all_tasks = []
    while task is not None:
        all_tasks.append(task)
        task = await benchmark_manager.step("hle")

    # ------------------------------------------------------------------
    # Filter tasks based on resume / filter mode
    # ------------------------------------------------------------------
    tasks_to_run = apply_filter(all_tasks, prev_results, prev_by_id, completed_ids, resume_file, filter_mode, result_saver)

    total_tasks = len(all_tasks)
    result_saver.update_total_tasks(total_tasks)
    mode = "bus" if use_bus else "direct LLM"
    logger.info(f"| Collected {total_tasks} HLE tasks ({len(tasks_to_run)} to run). Mode={mode}, concurrency={max_concurrency}...")

    semaphore = asyncio.Semaphore(max_concurrency)
    preloaded = total_tasks - len(tasks_to_run)
    completed_count_ref = [preloaded]   # start from already-done count
    completed_lock = asyncio.Lock()

    if use_bus:
        coros = [
            process_task_bus(
                t, semaphore, max_rounds, result_saver,
                total_tasks, completed_count_ref, completed_lock,
            )
            for t in tasks_to_run
        ]
    else:
        coros = [
            process_task_direct(
                t, semaphore, model_name, result_saver,
                total_tasks, completed_count_ref, completed_lock,
            )
            for t in tasks_to_run
        ]

    await asyncio.gather(*coros, return_exceptions=True)

    # Final stats
    logger.info(f"| {'='*50}")
    logger.info("| Final Statistics")
    logger.info(f"| {'='*50}")
    stats = await benchmark_manager.stats("hle")
    if stats:
        attempted = stats.correct + stats.wrong
        logger.info(f"| Overall: {attempted}/{stats.total} | Accuracy: {stats.accuracy:.2%}")
        logger.info(f"| Correct: {stats.correct} | Wrong: {stats.wrong}")
        logger.info(f"| Avg time: {stats.average_time:.2f}s")
        if stats.extra:
            logger.info(f"| Multiple-choice accuracy: {stats.extra.get('multiple_choice_accuracy', 0):.2%} "
                        f"({stats.extra.get('multiple_choice_total', 0)} tasks)")
            logger.info(f"| Exact-match accuracy:     {stats.extra.get('exact_match_accuracy', 0):.2%} "
                        f"({stats.extra.get('exact_match_total', 0)} tasks)")
        stats_path = os.path.join(save_dir, "stats.json")
        stats_data = stats.model_dump()
        stats_data["tasks"] = result_saver.results_data["results"]
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=4, ensure_ascii=False)
        logger.info(f"| Stats saved to: {stats_path}")


class LLMEvalResult(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning for whether the predicted answer is correct.")
    correct: bool = Field(description="True if the predicted answer is semantically equivalent to the ground truth.")

async def _eval_one(
    record: dict,
    model_name: str,
    semaphore: asyncio.Semaphore,
    completed_ref: list,
    total: int,
    lock: asyncio.Lock,
    correct_ref: list = None,
    wrong_ref: list = None,
) -> dict:
    task_id = record.get("task_id", "")
    gt = record.get("ground_truth", "")
    pred = record.get("predicted_answer", "")

    if not gt or not pred:
        return record

    async with semaphore:
        try:
            prompt = (
                f"You are an expert answer evaluator for the HLE (Humanity's Last Exam) benchmark.\n\n"
                f"Question:\n{record.get('task_input', '')}\n\n"
                f"Ground Truth Answer: {gt}\n\n"
                f"Predicted Answer: {pred}\n\n"
                f"Determine whether the predicted answer is correct. "
                f"Two answers are correct if they are semantically equivalent, "
                f"mathematically equal, or express the same concept "
                f"(e.g. '1/2' and '\\frac{{1}}{{2}}' are the same). "
                f"Be strict: do not accept partial answers or answers with extra incorrect content."
            )
            resp = await asyncio.wait_for(
                model_manager(
                    model=model_name,
                    messages=[HumanMessage(content=prompt)],
                    response_format=LLMEvalResult,
                ),
                timeout=60.0,
            )
            result: LLMEvalResult = resp.extra.parsed_model if resp and resp.extra else None
            if result:
                record["correct"] = result.correct
                record["llm_eval_reasoning"] = result.reasoning
                tag = "✅" if result.correct else "❌"
                logger.info(f"| {tag} [{task_id}] LLM eval: {result.correct}")
        except asyncio.TimeoutError:
            logger.warning(f"| [{task_id}] LLM eval timeout (60s)")
        except Exception as exc:
            logger.warning(f"| [{task_id}] LLM eval failed: {exc}")

        async with lock:
            completed_ref[0] += 1
            if correct_ref is not None and wrong_ref is not None:
                if record.get("correct"):
                    correct_ref[0] += 1
                else:
                    wrong_ref[0] += 1
                logger.info(f"| Eval progress: {completed_ref[0]}/{total} (✅ {correct_ref[0]} / ❌ {wrong_ref[0]})")
            else:
                logger.info(f"| Eval progress: {completed_ref[0]}/{total}")

    return record


async def eval(model_name: str, resume_file: str, max_concurrency: int = 8) -> None:
    if not resume_file or not os.path.exists(resume_file):
        logger.error(f"| eval: file not found: {resume_file}")
        return
    if not model_name:
        logger.error("| eval: --eval-model-name is required")
        return

    with open(resume_file, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    total = len(results)
    to_eval = [r for r in results if not r.get("correct", False)]
    logger.info(f"| LLM eval: {len(to_eval)}/{total} wrong results to re-evaluate using {model_name}")

    semaphore = asyncio.Semaphore(max_concurrency)
    completed_ref = [0]
    correct_ref = [0]
    wrong_ref = [0]
    lock = asyncio.Lock()

    await asyncio.gather(*[
        _eval_one(r, model_name, semaphore, completed_ref, len(to_eval), lock, correct_ref, wrong_ref)
        for r in to_eval
    ])

    # _eval_one mutates records in-place, so `results` already has updated values
    updated = results

    correct = sum(1 for r in updated if r.get("correct", False))
    data["results"] = sorted(updated, key=lambda r: r.get("task_id", ""))
    data["summary"].update({
        "completed_tasks": total,
        "correct_answers": correct,
        "accuracy": correct / total if total else 0.0,
        "last_updated": __import__("datetime").datetime.now().isoformat() + "Z",
        "eval_model": model_name,
    })

    with open(resume_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())

    logger.info(f"| LLM eval done: {correct}/{total} correct ({correct/total:.2%}) — saved to {resume_file}")


async def main():
    args = parse_args()

    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
    logger.info(f"| Config: {config.pretty_text}")

    model_name = args.model_name or config.model_name

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

    logger.info("| Initializing benchmark manager (HLE)...")
    hle_benchmark_config = config.hle_benchmark
    hle_benchmark_config.update({"start": args.start, "end": args.end})
    await benchmark_manager.initialize(benchmark_names=["hle"])

    result_saver = BenchmarkResultSaver("hle", args.max_concurrency, 0, model_name)
    logger.info(f"| Results will be saved to: {result_saver.get_file_path()}")

    if args.filter and not args.resume:
        logger.warning("| --filter has no effect without --resume, ignoring")

    # Resolve resume file path
    resume_file = None
    if args.resume:
        results_dir = os.path.join(config.workdir, "results", "hle")
        candidates = sorted(Path(results_dir).glob("benchmark_hle_*.json")) if os.path.isdir(results_dir) else []
        if not candidates:
            logger.error(f"| --resume: no previous results found in {results_dir}")
            return
        resume_file = str(candidates[-1])
        logger.info(f"| --resume: using {resume_file}")

    if args.eval_only:
        logger.info("| Running in evaluation-only mode (no training)...")
        await eval(model_name=args.eval_model_name, resume_file=resume_file)
    else:
        await run(
            max_concurrency=args.max_concurrency,
            max_rounds=args.max_rounds,
            result_saver=result_saver,
            use_bus=args.use_bus,
            model_name=model_name,
            resume_file=resume_file,
            filter_mode=(args.filter if args.filter != "none" else None) if args.resume else None,
        )

        if args.use_bus:
            from src.interaction import bus
            await bus.shutdown()

        await benchmark_manager.cleanup()
        logger.info("| Done.")


if __name__ == "__main__":
    asyncio.run(main())