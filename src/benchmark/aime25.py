import re
from typing import Optional, Any
from .types import Benchmark
from src.registry import BENCHMARK

@BENCHMARK.register_module(force=True)
class AIME25Benchmark(Benchmark):
    """
    AIME 2025 Benchmark implementation
    """
    name: str = "aime25"
    path: str = "datasets/AIME25"
    def _instantiate_dataset(self) -> Any:
        from src.data.aime25 import AIME25Dataset
        return AIME25Dataset(
            path=self.path,
            name=self.subset if self.subset else "all",
            split=self.split
        )

    def get_task_description(self) -> str:
        return "Please solve the following math contest problem. Put the final answer in \\boxed{}."

    async def _eval_logic(self, prediction: str, ground_truth: str, **kwargs) -> float:
        extracted_pred = self._extract_answer(prediction)
        clean_gt = self._clean_number(ground_truth)
        
        if extracted_pred is None:
            return 0.0
        
        return 1.0 if extracted_pred == clean_gt else 0.0

    def _clean_number(self, text: str) -> str:
        if not text: return ""
        text = text.strip().replace(",", "").replace("$", "")
        try:
            return str(int(float(text)))
        except (ValueError, TypeError):
            return text

    def _extract_answer(self, prediction: str) -> Optional[str]:
        if not prediction: return None
        
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        matches = re.findall(boxed_pattern, prediction)
        if matches:
            return self._clean_number(matches[-1])
            
        prompt_pattern = r"(?i)Answer:\s*(?:\$|\\\$)?\s*(-?[\d,]+)"
        matches = re.findall(prompt_pattern, prediction)
        if matches:
            return self._clean_number(matches[-1])
            
        numbers = re.findall(r"-?\d+", prediction)
        if numbers:
            return self._clean_number(numbers[-1])
            
        return None
