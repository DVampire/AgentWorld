import os
from typing import Optional, Any, List, Dict

import pandas as pd
from pydantic import Field, ConfigDict, PrivateAttr

from src.benchmark.types import Benchmark, Task, Stats
from src.registry import BENCHMARK
from src.utils import dedent, is_same, assemble_project_path

SYSTEM_PROMPT = dedent("""
    You are a helpful assistant that solves challenging benchmark questions with tools when needed.

    Return only the final answer, concise and with no explanation.
    If the answer is a number, return just the number unless the question explicitly requires a unit or format.
    If the answer is multiple choice, return only the option letter.
""")


@BENCHMARK.register_module(force=True)
class GAIABenchmark(Benchmark):
    """GAIA benchmark backed by a local Parquet dataset checkout."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="gaia", description="The name of the benchmark")
    path: str = Field(default="datasets/GAIA", description="Local path to the GAIA dataset root")
    year: str = Field(default="2023", description="Dataset year/config prefix")
    split: str = Field(default="validation", description="Dataset split")
    level: str = Field(default="all", description="Dataset level selector: all|1|2|3")
    system_prompt: Optional[str] = Field(default=SYSTEM_PROMPT, description="System prompt for GAIA tasks")

    _data_records: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _index: int = PrivateAttr(default=0)
    _tasks: List[Task] = PrivateAttr(default_factory=list)

    def __init__(self, base_dir: Optional[str] = None, start: Optional[int] = None, end: Optional[int] = None, **kwargs):
        super().__init__(base_dir=base_dir, start=start, end=end, **kwargs)
        os.makedirs(self.base_dir, exist_ok=True)

    def _dataset_root(self) -> str:
        return assemble_project_path(self.path)

    def _metadata_path(self) -> str:
        suffix = "" if self.level == "all" else f".level{self.level}"
        return os.path.join(self._dataset_root(), self.year, self.split, f"metadata{suffix}.parquet")

    async def initialize(self):
        metadata_path = self._metadata_path()
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"GAIA metadata file not found: {metadata_path}")

        try:
            df = pd.read_parquet(metadata_path)
        except ImportError as exc:
            raise ImportError(
                "GAIA benchmark requires parquet support. Install 'pyarrow' or "
                "'fastparquet' in the runtime environment."
            ) from exc
        records = df.to_dict(orient="records")
        self._data_records = self._apply_slice(records)
        await self.reset()

    async def reset(self) -> Optional[Task]:
        self._index = 0
        self._tasks = []
        return await self.step()

    async def step(self) -> Optional[Task]:
        if self._index >= len(self._data_records):
            return None

        record = self._data_records[self._index]
        self._index += 1

        task_id = str(record.get("task_id", f"{self._index:04d}"))
        level = str(record.get("Level", ""))
        question_text = str(record.get("Question", "")).strip()
        ground_truth = record.get("Final answer", "?")
        file_name = str(record.get("file_name", "") or "").strip()
        file_path = str(record.get("file_path", "") or "").strip()

        attachment_path = ""
        if file_path:
            resolved = os.path.join(self._dataset_root(), file_path)
            if os.path.exists(resolved):
                attachment_path = resolved
        elif file_name:
            resolved = os.path.join(self._dataset_root(), self.year, self.split, file_name)
            if os.path.exists(resolved):
                attachment_path = resolved

        extra = {
            "level": level,
            "file_name": file_name,
            "file_path": file_path,
            "attachment_path": attachment_path,
            "annotator_metadata": record.get("Annotator Metadata"),
            "split": self.split,
            "year": self.year,
        }

        return Task(
            task_id=task_id,
            input=question_text,
            system_prompt=self.system_prompt,
            ground_truth=ground_truth,
            extra=extra,
        )

    async def eval(self, task: Task) -> Optional[Task]:
        result = str(task.result).strip() if task.result is not None else ""
        ground_truth = str(task.ground_truth).strip() if task.ground_truth is not None else ""

        task.result = result
        task.ground_truth = ground_truth

        unscored = not ground_truth or ground_truth == "?"
        if unscored:
            task.score = None
        else:
            task.score = 1.0 if result and is_same(result, ground_truth) else 0.0

        self._tasks.append(task)
        return task

    async def stats(self) -> Optional[Stats]:
        total = len(self._data_records)
        attempted = len(self._tasks)
        scored_tasks = [task for task in self._tasks if task.score is not None]
        unscored_tasks = attempted - len(scored_tasks)
        correct = sum(1 for task in scored_tasks if task.score and task.score >= 1.0)
        wrong = len(scored_tasks) - correct

        task_times = {task.task_id: task.time for task in self._tasks if task.time is not None}
        average_time = sum(task_times.values()) / len(task_times) if task_times else 0.0

        per_level_accuracy: Dict[str, float] = {}
        per_level_total: Dict[str, int] = {}
        for level in ["1", "2", "3"]:
            level_tasks = [
                task for task in scored_tasks
                if str((task.extra or {}).get("level", "")) == level
            ]
            per_level_total[level] = len(level_tasks)
            per_level_accuracy[level] = (
                sum(1 for task in level_tasks if task.score and task.score >= 1.0) / len(level_tasks)
                if level_tasks else 0.0
            )

        return Stats(
            accuracy=correct / len(scored_tasks) if scored_tasks else 0.0,
            total=total,
            correct=correct,
            wrong=wrong,
            times=task_times,
            average_time=average_time,
            extra={
                "attempted": attempted,
                "scored_tasks": len(scored_tasks),
                "unscored_tasks": unscored_tasks,
                "per_level_accuracy": per_level_accuracy,
                "per_level_total": per_level_total,
                "split": self.split,
                "level": self.level,
                "year": self.year,
            },
        )
