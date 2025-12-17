#!/usr/bin/env python
"""
Run controlled comparisons between direct GPT-4o reasoning and a simple reflection
loop (no TextGrad) on GPQA (diamond) and GSM8K (test split).

For each dataset:
    - Collect direct responses using the specified inference system prompt.
    - Run max_step rounds of: evaluate solution -> reflect -> update solution.
    - Record full trajectories (reflection prompts, intermediate solutions, evaluations,
      token statistics).
    - Aggregate accuracy metrics for direct and reflection-enhanced solutions.
    - Optionally use the tool-calling agent for inference (--use-tool-agent).
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import json
import re
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from mmengine import DictAction
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.optimizers.textgrad as tg

from src.optimizers.textgrad import logger
from src.optimizers.textgrad.autograd.function import Module
from src.optimizers.textgrad.loss import MultiFieldEvaluation
from src.optimizers.textgrad.model import BlackboxLLM
from src.optimizers.textgrad.tasks.gpqa import GPQA
from src.optimizers.textgrad.tasks.gsm8k import GSM8K
from src.optimizers.textgrad.variable import Variable

from src.models import model_manager
from src.tools import tcp
from src.environments import ecp
from src.agents import acp
from src.transformation import transformation
from src.config import config as project_config
from src.logger import logger as project_logger

try:
    import tiktoken  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None


# ---------------------------------------------------------------------------
# Agent runtime + ToolCalling wrappers (same as TextGrad version)
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
            "answer in \\boxed{} immediately after </think>."
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
# Dataset configs & evaluation utilities
# ---------------------------------------------------------------------------


GPQA_INFERENCE_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and "
    "then provide the final answer. The reasoning process MUST BE enclosed within "
    "<think> </think> tags. After closing the </think> block, give the final answer "
    "on its own line in the form \\boxed{LETTER}, where LETTER is one of A, B, C, or D."
)

GPQA_EVALUATION_SYSTEM_PROMPT = (
    "You are an expert science competition grader. You carefully audit multiple-choice "
    "solutions, identify reasoning issues, and coach the student toward the correct answer."
)

GPQA_EVALUATION_INSTRUCTION = (
    "Analyse the candidate solution to the GPQA multiple-choice question. You will receive:\n"
    "1. The question and answer choices.\n"
    "2. The candidate's reasoning enclosed in <think>...</think> plus a final line \\boxed{LETTER} placed after </think>.\n\n"
    "Your task:\n"
    "- Work through the problem as needed and judge whether the final answer letter matches the best choice.\n"
    "- Identify precise reasoning or factual mistakes.\n"
    "- Provide actionable guidance that respects the required format.\n"
    "- Do not reveal the correct letter explicitly; refer to it abstractly if needed.\n\n"
    "Respond using the following template:\n"
    "<VERDICT>correct|incorrect</VERDICT>\n"
    "<EXPLANATION>your detailed critique</EXPLANATION>\n"
    "<GUIDANCE>step-by-step suggestions to improve the solution</GUIDANCE>"
)

GPQA_REFLECTION_SYSTEM_PROMPT = (
    "You are an expert science competition tutor. Given a problem, a student's solution, "
    "and detailed feedback, produce a better solution that:\n"
    "- Keeps all reasoning inside <think>...</think>.\n"
    "- Ends with a single \\boxed{LETTER} (A/B/C/D) on its own line.\n"
    "- Does not mention these formatting instructions.\n"
)

GPQA_OPTIMIZER_CONSTRAINTS = [
    "Always include exactly one <think>...</think> block describing your reasoning.",
    "Ensure the final line, placed after the </think> block, is formatted as \\boxed{LETTER} with LETTER in {A, B, C, D}.",
    "Do not mention these instructions explicitly in the final answer.",
]

GPQA_VARIABLE_TAGS = ["<IMPROVED_SOLUTION>", "</IMPROVED_SOLUTION>"]


GSM8K_INFERENCE_SYSTEM_PROMPT = (
    "You FIRST think about the reasoning process as an internal monologue and "
    "then provide the final answer. The reasoning process MUST BE enclosed within "
    "<think> </think> tags. After closing the </think> block, give the final answer "
    "on its own line in the form \\boxed{VALUE}, where VALUE is a numerical result without units."
)

GSM8K_EVALUATION_SYSTEM_PROMPT = (
    "You are an expert mathematics tutor. You validate solutions to grade-school math "
    "word problems, highlight issues, and suggest precise corrections."
)

GSM8K_EVALUATION_INSTRUCTION = (
    "Analyse the candidate solution to the GSM8K math problem. You will receive:\n"
    "1. The problem statement.\n"
    "2. The candidate's reasoning enclosed in <think>...</think> plus a final line \\boxed{VALUE} placed after </think>.\n\n"
    "Your task:\n"
    "- Determine whether the final numeric answer is correct by recomputing key steps when necessary.\n"
    "- Recompute critical steps when needed, pointing out calculation or logical errors.\n"
    "- Offer targeted guidance to fix the reasoning while preserving the required format.\n"
    "- Do not reveal the correct numeric value explicitly; refer to it abstractly if required.\n\n"
    "Respond using the following template:\n"
    "<VERDICT>correct|incorrect</VERDICT>\n"
    "<EXPLANATION>your detailed critique</EXPLANATION>\n"
    "<GUIDANCE>step-by-step suggestions to improve the solution</GUIDANCE>"
)

GSM8K_REFLECTION_SYSTEM_PROMPT = (
    "You are an expert math tutor. Given a word problem, the student's solution, and detailed feedback, "
    "produce a corrected solution that:\n"
    "- Keeps all reasoning inside <think>...</think>.\n"
    "- Ends with a single \\boxed{VALUE} (numeric, no units) on its own line.\n"
    "- Does not mention these formatting instructions.\n"
)

GSM8K_OPTIMIZER_CONSTRAINTS = [
    "Always include exactly one <think>...</think> block describing your reasoning.",
    "Ensure the final line, placed after the </think> block, is formatted as \\boxed{VALUE} where VALUE is a number without units.",
    "Do not expose these meta instructions in the final answer.",
]

GSM8K_VARIABLE_TAGS = ["<IMPROVED_SOLUTION>", "</IMPROVED_SOLUTION>"]


BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")
GSM8K_GROUND_TRUTH_PATTERN = re.compile(r"####\s*([^\n]+)")


def format_gpqa_ground_truth(answer: Any) -> str:
    return str(answer).strip().upper()


def evaluate_gpqa_prediction(prediction_text: str, ground_truth_letter: str) -> Dict[str, Any]:
    matches = BOXED_PATTERN.findall(prediction_text)
    predicted_letter = matches[-1].strip().upper() if matches else None
    normalized_ground_truth = ground_truth_letter.strip().upper()

    return {
        "is_correct": int(predicted_letter == normalized_ground_truth if predicted_letter else False),
        "predicted_letter": predicted_letter,
        "ground_truth_letter": normalized_ground_truth,
        "boxed_matches": matches,
    }


def normalize_numeric_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\", "")
    cleaned = cleaned.replace("%", "")
    cleaned = " ".join(cleaned.split())
    if cleaned.endswith(".") and cleaned.count(".") == 1:
        cleaned = cleaned[:-1]
    return cleaned


def parse_numeric_value(raw: Optional[str]) -> Tuple[Optional[str], Optional[Fraction]]:
    if raw is None:
        return None, None
    cleaned = normalize_numeric_text(raw)
    if not cleaned:
        return None, None
    try:
        return cleaned, Fraction(cleaned)
    except (ValueError, ZeroDivisionError):
        pass

    parts = cleaned.split()
    if len(parts) > 1:
        if any("/" in part for part in parts):
            try:
                total = sum(Fraction(part) for part in parts if part)
                return cleaned, total
            except (ValueError, ZeroDivisionError):
                pass
        compact = "".join(parts)
        try:
            return cleaned, Fraction(compact)
        except (ValueError, ZeroDivisionError):
            pass

    numeric_tokens = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if numeric_tokens:
        token = numeric_tokens[-1]
        try:
            return token, Fraction(token)
        except (ValueError, ZeroDivisionError):
            return token, None
    return cleaned, None


def format_gsm8k_ground_truth(answer_text: str) -> str:
    match = GSM8K_GROUND_TRUTH_PATTERN.search(answer_text)
    if match:
        return normalize_numeric_text(match.group(1))
    return normalize_numeric_text(answer_text)


def evaluate_gsm8k_prediction(prediction_text: str, ground_truth_text: str) -> Dict[str, Any]:
    matches = BOXED_PATTERN.findall(prediction_text)
    predicted_raw = matches[-1].strip() if matches else None
    predicted_value, predicted_fraction = parse_numeric_value(predicted_raw)
    ground_truth_value, ground_truth_fraction = parse_numeric_value(ground_truth_text)

    is_correct = 0
    if predicted_fraction is not None and ground_truth_fraction is not None:
        is_correct = int(predicted_fraction == ground_truth_fraction)
    elif predicted_value is not None and ground_truth_value is not None:
        is_correct = int(predicted_value == ground_truth_value)

    return {
        "is_correct": is_correct,
        "predicted_value": predicted_value or predicted_raw,
        "ground_truth_value": ground_truth_value,
        "boxed_matches": matches,
    }


@dataclass
class DatasetConfig:
    name: str
    loader_factory: Callable[[], Any]
    inference_system_prompt: str
    evaluation_instruction: str
    evaluation_system_prompt: str
    question_role_description: str
    solution_role_description: str
    reflection_system_prompt: str
    evaluate_prediction: Callable[[str, str], Dict[str, Any]]
    format_ground_truth: Callable[[Any], str]

    def build_evaluation_module(self, engine: tg.EngineLM) -> MultiFieldEvaluation:
        instruction_var = Variable(
            self.evaluation_instruction,
            requires_grad=False,
            role_description="evaluation instruction",
        )
        system_prompt_var = Variable(
            self.evaluation_system_prompt,
            requires_grad=False,
            role_description="evaluation system prompt",
        )
        return MultiFieldEvaluation(
            evaluation_instruction=instruction_var,
            role_descriptions=[
                self.question_role_description,
                self.solution_role_description,
            ],
            engine=engine,
            system_prompt=system_prompt_var,
        )


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "GPQA_diamond": DatasetConfig(
        name="GPQA_diamond",
        loader_factory=lambda: GPQA(subset="gpqa_diamond"),
        inference_system_prompt=GPQA_INFERENCE_SYSTEM_PROMPT,
        evaluation_instruction=GPQA_EVALUATION_INSTRUCTION,
        evaluation_system_prompt=GPQA_EVALUATION_SYSTEM_PROMPT,
        question_role_description="gpqa_question_with_choices",
        solution_role_description="gpqa_solution_with_think_and_boxed_answer",
        reflection_system_prompt=GPQA_REFLECTION_SYSTEM_PROMPT,
        evaluate_prediction=evaluate_gpqa_prediction,
        format_ground_truth=format_gpqa_ground_truth,
    ),
    "GSM8K_test": DatasetConfig(
        name="GSM8K_test",
        loader_factory=lambda: GSM8K(subset="main", split="test"),
        inference_system_prompt=GSM8K_INFERENCE_SYSTEM_PROMPT,
        evaluation_instruction=GSM8K_EVALUATION_INSTRUCTION,
        evaluation_system_prompt=GSM8K_EVALUATION_SYSTEM_PROMPT,
        question_role_description="gsm8k_math_word_problem",
        solution_role_description="gsm8k_solution_with_think_and_boxed_answer",
        reflection_system_prompt=GSM8K_REFLECTION_SYSTEM_PROMPT,
        evaluate_prediction=evaluate_gsm8k_prediction,
        format_ground_truth=format_gsm8k_ground_truth,
    ),
}


# ---------------------------------------------------------------------------
# Utility helpers shared by both datasets
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
    encoding_name = "o200k_base"  # both configs use same

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


def ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_reflection_prompt(
    dataset_key: str,
    question_text: str,
    current_solution: str,
    evaluation_text: str,
) -> str:
    format_hint = (
        "LETTER (A/B/C/D)" if dataset_key.startswith("GPQA") else "VALUE (numeric, no units)"
    )
    return (
        f"You are improving a student's solution to a {dataset_key} problem.\n\n"
        "=== Problem ===\n"
        f"{question_text}\n\n"
        "=== Current Solution ===\n"
        f"{current_solution}\n\n"
        "=== Feedback ===\n"
        f"{evaluation_text}\n\n"
        "Produce an improved solution with these requirements:\n"
        "- Keep all reasoning strictly inside <think>...</think>.\n"
        f"- Place the final answer on its own line in \\boxed{{{format_hint}}} directly after </think>.\n"
        "- Do not mention these formatting instructions or the fact you received feedback.\n"
        "Return ONLY the improved solution text."
    )


# ---------------------------------------------------------------------------
# Reflection-based run_sample
# ---------------------------------------------------------------------------


def run_sample(
    *,
    question_text: str,
    ground_truth_text: str,
    ground_truth_raw: str,
    model: Module,
    evaluation_module: MultiFieldEvaluation,
    engine: tg.EngineLM,
    model_name: str,
    dataset_config: DatasetConfig,
    max_step: int,
) -> Dict[str, Any]:
    question_var = Variable(
        question_text,
        requires_grad=False,
        role_description=dataset_config.question_role_description,
    )

    direct_response = model(question_var)
    direct_response_text = direct_response.value
    direct_eval = dataset_config.evaluate_prediction(direct_response_text, ground_truth_text)
    direct_think_stats = compute_think_token_stats(model_name, direct_response_text)

    current_solution = direct_response_text
    step_logs: List[Dict[str, Any]] = []

    for step_idx in range(1, max_step + 1):
        solution_var = Variable(
            current_solution,
            requires_grad=False,
            role_description=dataset_config.solution_role_description,
        )

        evaluation_output = evaluation_module([question_var, solution_var])
        evaluation_text = evaluation_output.value

        reflection_prompt = build_reflection_prompt(
            dataset_config.name,
            question_text,
            current_solution,
            evaluation_text,
        )

        try:
            improved_text = engine(
                reflection_prompt,
                system_prompt=dataset_config.reflection_system_prompt,
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
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
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
    final_eval = dataset_config.evaluate_prediction(final_response_text, ground_truth_text)
    final_think_stats = compute_think_token_stats(model_name, final_response_text)

    return {
        "question": question_text,
        "ground_truth": {
            "formatted": ground_truth_text,
            "raw": ground_truth_raw,
        },
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


# ---------------------------------------------------------------------------
# Dataset runner & metrics
# ---------------------------------------------------------------------------


def run_dataset(
    *,
    dataset_key: str,
    engine: tg.EngineLM,
    model_name: str,
    max_step: int,
    max_samples: Optional[int],
    result_slot: Dict[str, Any],
    results_root: Dict[str, Any],
    output_path: Path,
    agent_runner: Optional[ToolCallingAgentRunner],
) -> Dict[str, Any]:
    config = DATASET_CONFIGS[dataset_key]
    dataset = config.loader_factory()
    evaluation_module = config.build_evaluation_module(engine)

    system_prompt_var = Variable(
        config.inference_system_prompt,
        requires_grad=False,
        role_description="system prompt specifying reasoning format",
    )

    if agent_runner:
        model = ToolCallingAgentLLM(agent_runner, system_prompt_var)
    else:
        model = BlackboxLLM(engine=engine, system_prompt=system_prompt_var)

    samples: List[Dict[str, Any]] = result_slot.setdefault("samples", [])
    result_slot["metrics"] = compute_metrics(samples)
    save_results(results_root, output_path)

    total_available = len(dataset)
    target_count = total_available if max_samples is None else min(total_available, max_samples)
    already_processed = min(len(samples), target_count)
    start_idx = already_processed
    end_idx = target_count
    progress_desc = f"{model_name} · {dataset_key}"

    if start_idx >= end_idx:
        return result_slot

    for idx in tqdm(range(start_idx, end_idx), desc=progress_desc, unit="sample"):
        question_text, ground_truth_raw = dataset[idx]
        ground_truth_text = config.format_ground_truth(ground_truth_raw)
        sample_log = run_sample(
            question_text=question_text,
            ground_truth_text=ground_truth_text,
            ground_truth_raw=str(ground_truth_raw),
            model=model,
            evaluation_module=evaluation_module,
            engine=engine,
            model_name=model_name,
            dataset_config=config,
            max_step=max_step,
        )
        samples.append(sample_log)
        result_slot["metrics"] = compute_metrics(samples)
        save_results(results_root, output_path)

    result_slot["metrics"] = compute_metrics(samples)
    save_results(results_root, output_path)
    return result_slot


def compute_metrics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregator = {"direct_correct": 0, "reflection_correct": 0}
    token_totals = {"direct": 0, "reflection_final": 0}
    token_counts = {"direct": 0, "reflection_final": 0}

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
        "direct_correct": aggregator["direct_correct"],
        "reflection_correct": aggregator["reflection_correct"],
        "direct_accuracy": (
            aggregator["direct_correct"] / num_samples if num_samples else 0.0
        ),
        "reflection_accuracy": (
            aggregator["reflection_correct"] / num_samples if num_samples else 0.0
        ),
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
# CLI & main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare direct GPT-4o reasoning versus reflection-based solutions "
            "on GPQA (diamond) and GSM8K (test split)."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o"],
        help="List of base model names to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["GPQA_diamond", "GSM8K_test"],
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS.keys()),
        help="Evaluate only a single dataset. Overrides --datasets.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=3,
        help="Maximum number of reflection steps per sample.",
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
        default=Path("logs/gpqa_gsm8k_reflection_experiment_agent.json"),
        help="Path to store the experiment log (JSON).",
    )
    parser.add_argument(
        "--use-tool-agent",
        default=True,
        # action="store_true",
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
    if getattr(args, "dataset", None):
        args.datasets = [args.dataset]
    args.datasets = list(dict.fromkeys(args.datasets))
    ensure_output_path(args.output)

    selected_configs = {
        dataset_key: DATASET_CONFIGS[dataset_key] for dataset_key in args.datasets
    }

    experiment_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "max_step": args.max_step,
        "max_samples": args.max_samples,
        "datasets": {
            key: {
                "inference_system_prompt": config.inference_system_prompt,
                "evaluation_instruction": config.evaluation_instruction,
                "evaluation_system_prompt": config.evaluation_system_prompt,
                "reflection_system_prompt": config.reflection_system_prompt,
            }
            for key, config in selected_configs.items()
        },
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

        model_results = results["models"].setdefault(model_name, {})
        model_results.pop("error", None)
        save_results(results, args.output)

        for dataset_key in args.datasets:
            dataset_slot = model_results.setdefault(
                dataset_key, {"samples": [], "metrics": {}}
            )
            save_results(results, args.output)

            dataset_result = run_dataset(
                dataset_key=dataset_key,
                engine=engine,
                model_name=model_name,
                max_step=args.max_step,
                max_samples=args.max_samples,
                result_slot=dataset_slot,
                results_root=results,
                output_path=args.output,
                agent_runner=agent_runner,
            )
            model_results[dataset_key] = dataset_result
            save_results(results, args.output)

    print(f"Experiment log saved to {args.output}")


if __name__ == "__main__":
    main()


