import re
from typing import Optional, Any
from .types import Benchmark
from src.registry import BENCHMARK

@BENCHMARK.register_module(force=True)
class GSM8kBenchmark(Benchmark):
    """
    GSM8k Benchmark implementation
    """
    name: str = "gsm8k"
    path: str = "datasets/gsm8k"
    def _instantiate_dataset(self) -> Any:
        from src.data.gsm8k import GSM8kDataset
        return GSM8kDataset(
            path=self.path,
            name=self.subset if self.subset else "main",
            split=self.split
        )

    def get_task_description(self) -> str:
        return (
            "You will answer a mathematical reasoning question. Think step by step. "
            "The last line of your response should be of the following format: "
            "'Answer: $VALUE' where VALUE is a numerical value."
        )

    async def _eval_logic(self, prediction: str, ground_truth: str, **kwargs) -> float:
        extracted_pred = self._extract_answer(prediction)
        clean_gt = self._clean_number(ground_truth)
        
        if extracted_pred is None:
            return 0.0
        
        return 1.0 if extracted_pred == clean_gt else 0.0

    def _clean_number(self, text: str) -> str:
        if not text:
            return ""
        text = text.strip().replace(",", "").replace("$", "")
        try:
            return str(int(float(text)))
        except (ValueError, TypeError):
            return text

    def _extract_answer(self, prediction: str) -> Optional[str]:
        if not prediction:
            return None

        # 1. Answer: $VALUE
        prompt_pattern = r"(?i)Answer:\s*(?:\$|\\\$)?\s*(-?[\d,]+)"
        matches = re.findall(prompt_pattern, prediction)
        if matches:
            return self._clean_number(matches[-1])

        # 2. \boxed{VALUE}
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        matches = re.findall(boxed_pattern, prediction)
        if matches:
            return self._clean_number(matches[-1])

        # 3. #### VALUE
        if "####" in prediction:
            return self._clean_number(prediction.split("####")[-1])

        # 4. Last number
        numbers = re.findall(r"-?\d+", prediction)
        if numbers:
            return self._clean_number(numbers[-1])

        return None

