import re
from typing import Optional, Any, List, Dict
from pydantic import Field, ConfigDict, PrivateAttr


from src.benchmark.types import Benchmark, Task, Stats
from src.registry import BENCHMARK
from src.benchmark.utils import clean_text
from src.utils import dedent

SYSTEM_PROMPT = dedent("""
    You are a helpful assistant that solves math contest problems. Please think step by step and provide the final answer.
    
    Example:
    Problem: If a + b = 10 and a - b = 4, what is the value of a^2 - b^2?
    Output format:
    The output should be a JSON object with the following fields, DO NOT add any other text like "```json" or "```" or anything else:
    {
        "reasoning": "Step 1: Solve for a and b\n\na + b = 10\n\na - b = 4\n\nAdding the two equations, we get 2a = 14, so a = 7.\n\tSubstituting a = 7 into the first equation, we get 7 + b = 10, so b = 3.\n\nStep 2: Calculate a^2 - b^2\n\ta^2 - b^2 = 7^2 - 3^2 = 49 - 9 = 40\n\nStep 3: Provide the final answer\n\tThe final answer is 40.",
        "answer": 40
    }
    
    Please solve the following problem:
""")
@BENCHMARK.register_module(force=True)
class GSM8kBenchmark(Benchmark):
    """
    GSM8k Benchmark implementation
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="gsm8k", description="The name of the benchmark")
    path: str = Field(default="datasets/gsm8k", description="The path to the benchmark dataset")
    
    _data_records: List[Dict] = PrivateAttr(default_factory=list)
    _index: int = PrivateAttr(default=0)
    _tasks: List[Task] = PrivateAttr(default_factory=list)
    
    system_prompt: Optional[str] = Field(default=SYSTEM_PROMPT, description="The system prompt for the benchmark")

    async def initialize(self):
        from src.data.gsm8k import GSM8kDataset
        dataset = GSM8kDataset(
            path=self.path,
            name=self.subset if self.subset else "main",
            split=self.split
        )
        if hasattr(dataset, 'data'):
            self._data_records = dataset.data.to_dict(orient="records")
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
        
        return Task(
            task_id=f"{self._index:04d}",
            input=record.get("question") or record.get("prompt") or "",
            system_prompt=self.system_prompt,
            ground_truth=record.get("true_answer") or record.get("answer"),
            extra={k: v for k, v in record.items() if k not in ["true_answer", "answer", "task_id", "id", "question", "prompt"]}
        )

    async def eval(self, task: Task) -> Optional[Task]:
        answer = str(task.answer) if task.answer is not None else ""
        ground_truth = str(task.ground_truth) if task.ground_truth is not None else ""
        
        clean_answer = clean_text(answer) if answer is not None else None
        clean_ground_truth = clean_text(ground_truth) if ground_truth is not None else None
        
        task.answer = clean_answer
        task.ground_truth = clean_ground_truth
        
        task.score = 1.0 if clean_answer == clean_ground_truth and clean_answer is not None else 0.0
        self._tasks.append(task)
        return task

    async def stats(self) -> Optional[Stats]:
        total = len(self._data_records)
        attempted = len(self._tasks)
        correct = sum(1 for r in self._tasks if r.score and r.score >= 1.0)
        
        task_times = {r.task_id: r.time for r in self._tasks if r.time is not None}
        avg_time = sum(task_times.values()) / len(task_times) if task_times else 0.0
        
        return Stats(
            accuracy=correct / attempted if attempted > 0 else 0.0,
            total=total,
            correct=correct,
            wrong=attempted - correct,
            times=task_times,
            average_time=avg_time
        )

