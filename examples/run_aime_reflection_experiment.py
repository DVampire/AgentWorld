#!/usr/bin/env python
"""
Run a controlled comparison between direct inference and simple reflection-based
optimization on the AIME24 and AIME25 datasets.

This script mirrors `run_aime_textgrad_experiment.py` but replaces TextGrad with
plain reflection loops:
    * Direct reasoning with a fixed system prompt.
    * max_step rounds of: evaluate solution → reflect → generate improved solution.
    * Logs full trajectories and aggregates accuracy/token statistics.
    * Optionally uses the tool-calling agent as the base model when --use-tool-agent is set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional

from dotenv import load_dotenv
from mmengine import DictAction
from tqdm import tqdm

import src.optimizers.textgrad as tg

from src.optimizers.textgrad import logger
from src.optimizers.textgrad.autograd.function import Module
from src.optimizers.textgrad.loss import MultiFieldEvaluation
from src.optimizers.textgrad.model import BlackboxLLM
from src.optimizers.textgrad.tasks.aime import AIME24, AIME25
from src.optimizers.textgrad.tasks.big_bench_hard import parse_integer_answer
from src.optimizers.textgrad.variable import Variable

from src.models import model_manager
from src.tools import tcp
from src.environments import ecp
from src.agents import acp
from src.transformation import transformation
from src.config import config as project_config
from src.logger import logger as project_logger
import asyncio
import threading
import atexit

try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


# ---------------------------------------------------------------------------
# Agent runtime and wrapper (same as run_aime_textgrad_experiment)
# ---------------------------------------------------------------------------
class AgentRuntime:
    """Run async agent operations on a dedicated event loop thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._closed = False
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Awaitable[Any]) -> Any:
        if self._closed:
            raise RuntimeError("AgentRuntime has been closed.")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
        self._closed = True


class ToolCallingAgentRunner:
    """Synchronous wrapper around the async tool-calling agent."""

    def __init__(self, agent: Any, runtime: AgentRuntime):
        self._agent = agent
        self._runtime = runtime
        self._closed = False

    def __call__(self, system_prompt: str, question_text: str) -> str:
        task = self._compose_task(system_prompt, question_text)
        result = self._runtime.run(self._agent.ainvoke(task=task, files=[]))
        return result if isinstance(result, str) else str(result)

    def close(self) -> None:
        if self._closed:
            return
        self._runtime.shutdown()
        self._closed = True

    @staticmethod
    def _compose_task(system_prompt: str, question_text: str) -> str:
        sections = []
        if system_prompt:
            sections.append(system_prompt.strip())
        sections.append("Problem:\n" + question_text.strip())
        sections.append(
            "Important: keep all reasoning inside <think>...</think> and place the final "
            "answer in \\boxed{VALUE} immediately after </think>."
        )
        return "\n\n".join(sections)


class ToolCallingAgentLLM(Module):
    """Minimal Module that proxies inference to the tool-calling agent."""

    def __init__(self, runner: ToolCallingAgentRunner, system_prompt: Variable):
        self.runner = runner
        self.system_prompt = system_prompt

    def parameters(self) -> List[Variable]:
        return [self.system_prompt] if self.system_prompt else []

    def forward(self, x: Variable) -> Variable:
        system_prompt_value = self.system_prompt.value if self.system_prompt else ""
        response_text = self.runner(system_prompt_value, x.value)
        return Variable(
            value=response_text,
            requires_grad=False,
            role_description="tool_calling_agent_response",
        )


async def _async_prepare_tool_agent(
    *,
    agent_name: str,
    config_path: Path,
    cfg_options: Optional[Dict[str, Any]],
) -> Any:
    args = argparse.Namespace(config=str(config_path), cfg_options=cfg_options)
    project_config.init_config(str(config_path), args)
    project_logger.init_logger(project_config)

    use_local_proxy = getattr(project_config, "use_local_proxy", False)
    await model_manager.initialize(use_local_proxy=use_local_proxy)

    env_names = list(getattr(project_config, "env_names", []) or [])
    tool_names = list(getattr(project_config, "tool_names", []) or [])
    agent_names = list(getattr(project_config, "agent_names", []) or [])

    if env_names:
        await ecp.initialize(env_names)
    if tool_names:
        await tcp.initialize(tool_names)
    if agent_names:
        await acp.initialize(agent_names)
    if env_names:
        await transformation.transform(type="e2t", env_names=env_names)

    agent_info = acp.get_info(agent_name)
    if agent_info is None or agent_info.instance is None:
        available = ", ".join(acp.list())
        raise ValueError(f"Agent '{agent_name}' not found. Available agents: {available}")
    return agent_info.instance


def build_tool_calling_agent_runner(
    *,
    agent_name: str,
    config_path: Path,
    cfg_options: Optional[Dict[str, Any]],
) -> ToolCallingAgentRunner:
    runtime = AgentRuntime()
    try:
        agent = runtime.run(
            _async_prepare_tool_agent(
                agent_name=agent_name,
                config_path=config_path,
                cfg_options=cfg_options,
            )
        )
    except Exception:
        runtime.shutdown()
        raise

    runner = ToolCallingAgentRunner(agent=agent, runtime=runtime)
    atexit.register(runner.close)
    return runner


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


INFERENCE_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and "
    "then provide the final answer. The reasoning process MUST BE enclosed within "
    "<think> </think> tags. The final answer MUST BE put in \\boxed{}."
)

EVALUATION_SYSTEM_PROMPT = (
    "You are an expert mathematics competition grader. You carefully analyse "
    "candidate solutions, identify issues, and provide actionable feedback that "
    "helps improve the answer while keeping the required output format."
)

EVALUATION_INSTRUCTION = (
    "Analyse the candidate solution to the contest problem. You will receive:\n"
    "1. The problem statement.\n"
    "2. The candidate's full reasoning enclosed in <think>...</think> plus a final "
    "answer in \\boxed{}.\n\n"
    "Your task:\n"
    "- Determine whether the boxed answer is logically correct based on the reasoning.\n"
    "- Recompute key steps when needed, but never reveal the exact numeric answer even "
    "if you know or infer it.\n"
    "- Point out precise reasoning or computation mistakes (if any).\n"
    "- Provide targeted guidance that helps amend the reasoning while keeping the "
    "<think> format intact.\n\n"
    "Respond using the following template:\n"
    "<VERDICT>correct|incorrect</VERDICT>\n"
    "<EXPLANATION>your detailed critique</EXPLANATION>\n"
    "<GUIDANCE>step-by-step instructions to improve the solution</GUIDANCE>"
)

REFLECTION_SYSTEM_PROMPT = (
    "You are an expert mathematics tutor. Given a problem, a student's solution, "
    "and detailed feedback, you must produce an improved solution.\n\n"
    "Requirements:\n"
    "- Keep all reasoning strictly inside <think>...</think>.\n"
    "- Put ONLY the final numeric answer in a single \\boxed{VALUE} expression on the last line.\n"
    "- Do not explain the meta-instructions.\n"
)

DATASET_REGISTRY = {
    "AIME24": AIME24,
    "AIME25": AIME25,
}

MODEL_TO_ENCODING = {
    "gpt-4o": "o200k_base",
    "gpt-5-mini": "o200k_base",
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def extract_think_block(text: str) -> str:
    start_tag = "<think>"
    end_tag = "</think>"
    start = text.find(start_tag)
    if start == -1:
        return ""
    end = text.find(end_tag, start + len(start_tag))
    if end == -1:
        return ""
    return text[start + len(start_tag) : end].strip()


def compute_think_token_stats(model_name: str, response_text: str) -> Dict[str, Any]:
    think_text = extract_think_block(response_text)
    encoding_name = MODEL_TO_ENCODING.get(model_name)

    stats: Dict[str, Any] = {
        "think_text": think_text,
        "token_count": 0,
        "mode": "missing_think" if not think_text else "unknown",
        "encoding": encoding_name,
    }

    if not think_text:
        return stats

    if tiktoken and encoding_name:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            stats["token_count"] = len(encoding.encode(think_text))
            stats["mode"] = "tiktoken"
            return stats
        except Exception:  # pragma: no cover - tokenizer edge case
            pass

    stats["token_count"] = len(think_text.split())
    stats["mode"] = "approx_word_count"
    return stats


def extract_final_answer(text: str) -> Optional[int]:
    tokens = [token for token in text.strip().split() if any(ch.isdigit() for ch in token)]
    if not tokens:
        return None
    candidate = tokens[-1].split(".")[0]
    digits = "".join(ch for ch in candidate if ch.isdigit())
    if not digits:
        return None
    return int(digits)


def evaluate_prediction(prediction_text: str, ground_truth_text: str) -> Dict[str, Any]:
    parsed_prediction = parse_integer_answer(prediction_text)
    parsed_ground_truth = parse_integer_answer(ground_truth_text)
    is_correct = int(
        (parsed_prediction is not None)
        and (parsed_ground_truth is not None)
        and (parsed_prediction == parsed_ground_truth)
    )
    return {
        "is_correct": is_correct,
        "parsed_prediction": parsed_prediction,
        "parsed_ground_truth": parsed_ground_truth,
        "extracted_prediction": extract_final_answer(prediction_text),
    }


def ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_evaluation_module(engine: tg.EngineLM) -> MultiFieldEvaluation:
    evaluation_instruction = Variable(
        EVALUATION_INSTRUCTION,
        requires_grad=False,
        role_description="instructions for evaluating a math contest solution",
    )
    evaluation_system_prompt = Variable(
        EVALUATION_SYSTEM_PROMPT,
        requires_grad=False,
        role_description="system prompt for the evaluation LLM",
    )
    return MultiFieldEvaluation(
        evaluation_instruction=evaluation_instruction,
        role_descriptions=[
            "contest problem statement",
            "candidate solution with reasoning and boxed answer",
        ],
        engine=engine,
        system_prompt=evaluation_system_prompt,
    )


# ---------------------------------------------------------------------------
# Reflection-based core logic
# ---------------------------------------------------------------------------


def build_reflection_prompt(
    question_text: str,
    current_solution: str,
    evaluation_text: str,
) -> str:
    """Build a prompt that asks the model to improve the solution based on feedback."""
    return (
        "You are improving a student's solution to a math contest problem.\n\n"
        "=== Problem ===\n"
        f"{question_text}\n\n"
        "=== Current Solution ===\n"
        f"{current_solution}\n\n"
        "=== Feedback ===\n"
        f"{evaluation_text}\n\n"
        "Please produce an improved solution that:\n"
        "- Keeps all reasoning inside <think>...</think>.\n"
        "- Puts ONLY the final numeric answer in a single \\boxed{VALUE} expression on the last line.\n"
        "- Does not discuss these formatting rules explicitly.\n"
        "Return ONLY the improved solution text."
    )


def run_sample(
    *,
    question_text: str,
    ground_truth_text: str,
    model: Module,  # BlackboxLLM or ToolCallingAgentLLM
    evaluation_module: MultiFieldEvaluation,
    engine: tg.EngineLM,
    model_name: str,
    max_step: int,
) -> Dict[str, Any]:
    question_var = Variable(
        question_text,
        requires_grad=False,
        role_description="contest problem statement",
    )

    # Direct inference
    direct_response = model(question_var)
    direct_response_text = direct_response.value
    direct_eval = evaluate_prediction(direct_response_text, ground_truth_text)
    direct_think_stats = compute_think_token_stats(model_name, direct_response_text)

    # Reflection loop
    current_solution = direct_response_text
    step_logs: List[Dict[str, Any]] = []

    for step_idx in range(1, max_step + 1):
        solution_var = Variable(
            current_solution,
            requires_grad=False,
            role_description="candidate solution with reasoning and boxed answer",
        )
        evaluation_output = evaluation_module([question_var, solution_var])
        evaluation_text = evaluation_output.value

        reflection_prompt = build_reflection_prompt(
            question_text=question_text,
            current_solution=current_solution,
            evaluation_text=evaluation_text,
        )

        try:
            improved_text = engine(
                reflection_prompt,
                system_prompt=REFLECTION_SYSTEM_PROMPT,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.error(f"Reflection step {step_idx} failed: {exc}")
            step_logs.append(
                {
                    "step": step_idx,
                    "error": f"reflection_failed: {exc}",
                    "evaluation_output": evaluation_text,
                    "reflection_prompt": reflection_prompt,
                }
            )
            break

        improved_text = str(improved_text).strip()
        if improved_text.startswith("```"):
            lines = improved_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            improved_text = "\n".join(lines).strip()

        current_solution = improved_text
        think_stats = compute_think_token_stats(model_name, current_solution)

        step_logs.append(
            {
                "step": step_idx,
                "evaluation_output": evaluation_text,
                "reflection_prompt": reflection_prompt,
                "improved_solution": current_solution,
                "think_token_stats": think_stats,
            }
        )

    final_response_text = current_solution
    final_eval = evaluate_prediction(final_response_text, ground_truth_text)
    final_think_stats = compute_think_token_stats(model_name, final_response_text)

    return {
        "question": question_text,
        "ground_truth": ground_truth_text,
        "direct": {
            "response": direct_response_text,
            "evaluation": direct_eval,
            "think_token_stats": direct_think_stats,
        },
        "reflection": {
            "initial_solution": direct_response_text,
            "steps": step_logs,
            "final_response": final_response_text,
            "final_evaluation": final_eval,
            "final_think_token_stats": final_think_stats,
        },
    }


def run_dataset(
    *,
    dataset_name: str,
    engine: tg.EngineLM,
    model: Module,
    model_name: str,
    max_step: int,
    max_samples: Optional[int],
    result_slot: Dict[str, Any],
    results_root: Dict[str, Any],
    output_path: Path,
) -> Dict[str, Any]:
    dataset_cls = DATASET_REGISTRY[dataset_name]
    dataset = dataset_cls(split="all")

    evaluation_module = build_evaluation_module(engine)

    samples: List[Dict[str, Any]] = result_slot.setdefault("samples", [])
    result_slot["metrics"] = compute_metrics(samples)
    save_results(results_root, output_path)

    total_available = len(dataset)
    target_count = total_available if max_samples is None else min(total_available, max_samples)
    already_processed = min(len(samples), target_count)
    start_idx = already_processed
    end_idx = target_count
    progress_desc = f"{model_name} · {dataset_name}"

    if start_idx >= end_idx:
        return result_slot

    for idx in tqdm(range(start_idx, end_idx), desc=progress_desc, unit="sample"):
        question_text, ground_truth_text = dataset[idx]
        sample_log = run_sample(
            question_text=question_text,
            ground_truth_text=str(ground_truth_text),
            model=model,
            evaluation_module=evaluation_module,
            engine=engine,
            model_name=model_name,
            max_step=max_step,
        )
        samples.append(sample_log)
        result_slot["metrics"] = compute_metrics(samples)
        save_results(results_root, output_path)

    result_slot["metrics"] = compute_metrics(samples)
    save_results(results_root, output_path)
    return result_slot


def compute_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregator = defaultdict(int)
    token_totals = defaultdict(int)
    token_counts = defaultdict(int)

    for sample in samples:
        if sample["direct"]["evaluation"]["is_correct"]:
            aggregator["direct_correct"] += 1
        if sample["reflection"]["final_evaluation"]["is_correct"]:
            aggregator["reflection_correct"] += 1

        direct_tokens = sample["direct"]["think_token_stats"]["token_count"]
        if direct_tokens:
            token_totals["direct"] += direct_tokens
            token_counts["direct"] += 1

        final_tokens = sample["reflection"]["final_think_token_stats"]["token_count"]
        if final_tokens:
            token_totals["reflection_final"] += final_tokens
            token_counts["reflection_final"] += 1

    num_samples = len(samples)
    metrics = {
        "num_samples": num_samples,
        "direct_accuracy": aggregator["direct_correct"] / num_samples if num_samples else 0.0,
        "reflection_accuracy": aggregator["reflection_correct"] / num_samples if num_samples else 0.0,
        "direct_correct": aggregator["direct_correct"],
        "reflection_correct": aggregator["reflection_correct"],
    }

    if token_counts["direct"]:
        metrics["avg_direct_think_tokens"] = (
            token_totals["direct"] / token_counts["direct"]
        )
    if token_counts["reflection_final"]:
        metrics["avg_reflection_think_tokens"] = (
            token_totals["reflection_final"] / token_counts["reflection_final"]
        )

    return metrics


def save_results(results: Dict[str, Any], path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct reasoning vs reflection-optimized solutions on AIME datasets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4.1"],
        help="List of base model names to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["AIME24", "AIME25"],
        choices=list(DATASET_REGISTRY.keys()),
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=3,
        help="Maximum number of reflection update steps per sample.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples per dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/aime_reflection_experiment_4.1.json"),
        help="Path to store the experiment log (JSON).",
    )
    parser.add_argument(
        "--use-tool-agent",
        action="store_true",
        help="Use the tool-calling agent for direct inference instead of the base model.",
    )
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=Path("configs/tool_calling_agent.py"),
        help="Config file for the tool-calling agent (used when --use-tool-agent is set).",
    )
    parser.add_argument(
        "--agent-name",
        default="tool_calling",
        help="Agent registered in ACP to use for inference (requires --use-tool-agent).",
    )
    parser.add_argument(
        "--agent-cfg-options",
        nargs="+",
        action=DictAction,
        help="Override options for the agent config, same format as mmengine DictAction.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(override=True)
    args = parse_args()
    ensure_output_path(args.output)

    experiment_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "system_prompt": INFERENCE_SYSTEM_PROMPT,
        "evaluation_instruction": EVALUATION_INSTRUCTION,
        "evaluation_system_prompt": EVALUATION_SYSTEM_PROMPT,
        "max_step": args.max_step,
        "max_samples": args.max_samples,
    }

    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            results: Dict[str, Any] = json.load(f)
    else:
        results = {}

    results.setdefault("meta", {})
    results["meta"].update(experiment_meta)
    results.setdefault("models", {})

    save_results(results, args.output)

    agent_runner: Optional[ToolCallingAgentRunner] = None
    if args.use_tool_agent:
        try:
            agent_runner = build_tool_calling_agent_runner(
                agent_name=args.agent_name,
                config_path=args.agent_config,
                cfg_options=args.agent_cfg_options,
            )
        except Exception as exc:
            print(f"[工具代理初始化失败] {exc}")
            sys.exit(1)

    for model_name in args.models:
        try:
            engine = tg.get_engine(model_name)
        except ValueError as exc:
            print(f"[跳过模型] {model_name}: {exc}")
            results["models"][model_name] = {"error": str(exc)}
            save_results(results, args.output)
            continue

        system_prompt_var = Variable(
            INFERENCE_SYSTEM_PROMPT,
            requires_grad=False,
            role_description="system prompt specifying reasoning format",
        )
        if agent_runner:
            model = ToolCallingAgentLLM(agent_runner, system_prompt_var)
        else:
            model = BlackboxLLM(engine=engine, system_prompt=system_prompt_var)

        model_results = results["models"].setdefault(model_name, {})
        model_results.pop("error", None)
        save_results(results, args.output)

        for dataset_name in args.datasets:
            dataset_slot = model_results.setdefault(
                dataset_name, {"samples": [], "metrics": {}}
            )
            save_results(results, args.output)

            dataset_result = run_dataset(
                dataset_name=dataset_name,
                engine=engine,
                model=model,
                model_name=model_name,
                max_step=args.max_step,
                max_samples=args.max_samples,
                result_slot=dataset_slot,
                results_root=results,
                output_path=args.output,
            )
            model_results[dataset_name] = dataset_result
            save_results(results, args.output)

    print(f"Experiment log saved to {args.output}")


if __name__ == "__main__":
    main()


