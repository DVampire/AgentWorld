import re
from typing import Optional, Any
from .types import Benchmark
from src.registry import BENCHMARK

@BENCHMARK.register_module(force=True)
class GPQABenchmark(Benchmark):
    """
    GPQA (Google-Proof Q&A) Benchmark implementation
    """
    name: str = "gpqa"
    path: str = "datasets/GPQA"
    def _instantiate_dataset(self) -> Any:
        from src.data.GPQA import GPQADataset
        return GPQADataset(
            path=self.path,
            name=self.subset if self.subset else "gpqa_diamond", 
            split=self.split
        )

    def get_task_description(self) -> str:
        return (
            "You are an expert scientist. Answer the following multiple-choice question. "
            "Select the correct answer from the given options. "
            "End your response with the format 'Answer: <letter>' (e.g., Answer: A)."
        )

    async def _eval_logic(self, prediction: str, ground_truth: str, **kwargs) -> float:
        extracted_pred = self._extract_answer(prediction)
        clean_gt = self._clean_choice(ground_truth)
        
        if extracted_pred is None:
            return 0.0
        
        return 1.0 if extracted_pred == clean_gt else 0.0

    def _clean_choice(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip().upper()
        text = text.replace("(", "").replace(")", "").replace(".", "")
        return text

    def _extract_answer(self, prediction: str) -> Optional[str]:
        if not prediction:
            return None

        prompt_pattern = r"(?i)Answer:\s*\(?([A-D])\)?"
        matches = re.findall(prompt_pattern, prediction)
        if matches:
            return self._clean_choice(matches[-1])

        boxed_pattern = r"\\boxed\{\s*([A-D])\s*\}"
        matches = re.findall(boxed_pattern, prediction)
        if matches:
            return self._clean_choice(matches[-1])

        paren_pattern = r"\s\(([A-D])\)" 
        matches = re.findall(paren_pattern, prediction)
        if matches:
            return self._clean_choice(matches[-1])

        loose_pattern = r"(?:^|\s)([A-D])(?:\.|,|$)"
        matches = re.findall(loose_pattern, prediction)
        if matches:
            return self._clean_choice(matches[-1])

        return None
