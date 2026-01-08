import re
from typing import Optional, Any, List, Dict
from pydantic import Field, ConfigDict, PrivateAttr


from src.benchmark.types import Benchmark, Task, Stats
from src.registry import BENCHMARK
from src.benchmark.utils import clean_text

SYSTEM_PROMPT = "You are an expert scientist. Answer the following multiple-choice question. Select the correct answer from the given options. Put the final answer in \\boxed{} (e.g., \\boxed{A})."

@BENCHMARK.register_module(force=True)
class GPQABenchmark(Benchmark):
    """
    GPQA (Google-Proof Q&A) Benchmark implementation
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="gpqa", description="The name of the benchmark")
    path: str = Field(default="datasets/GPQA", description="The path to the benchmark dataset")
    
    _data_records: List[Dict] = PrivateAttr(default_factory=list)
    _index: int = PrivateAttr(default=0)
    _tasks: List[Task] = PrivateAttr(default_factory=list)
    
    system_prompt: Optional[str] = Field(default=SYSTEM_PROMPT, description="The system prompt for the benchmark")

    async def initialize(self):
        from src.data.GPQA import GPQADataset
        dataset = GPQADataset(
            path=self.path,
            name=self.subset if self.subset else "gpqa_diamond", 
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
        prediction = str(task.prediction) if task.prediction is not None else ""
        ground_truth = str(task.ground_truth) if task.ground_truth is not None else ""
        
        # Extract answer from prediction (patterned after AIME)
        extracted_pred = None
        if prediction:
            # 1. Try \boxed{}
            boxed_matches = re.findall(r"\\boxed\{\s*([A-D])\s*\}", prediction)
            if boxed_matches:
                extracted_pred = boxed_matches[-1]
            else:
                # 2. Try Answer: <letter>
                answer_matches = re.findall(r"(?i)Answer:\s*\(?([A-D])\)?", prediction)
                if answer_matches:
                    extracted_pred = answer_matches[-1]
                else:
                    # 3. Try parenthesis pattern (A)
                    paren_matches = re.findall(r"\s\(([A-D])\)", prediction)
                    if paren_matches:
                        extracted_pred = paren_matches[-1]
                    else:
                        # 4. Try loose pattern A.
                        loose_matches = re.findall(r"(?:^|\s)([A-D])(?:\.|,|$)", prediction)
                        if loose_matches:
                            extracted_pred = loose_matches[-1]

        clean_pred = clean_text(extracted_pred) if extracted_pred is not None else None
        clean_gt = clean_text(ground_truth)
        
        task.score = 1.0 if clean_pred == clean_gt and clean_pred is not None else 0.0
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
