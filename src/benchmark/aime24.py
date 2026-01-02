# src/benchmark/implementations.py

import re
import asyncio
from typing import Optional, Any
from .types import Benchmark
from src.registry import DATASET

class AIMEBenchmark(Benchmark):
    """
    AIME 具体实现
    """
    async def _load_dataset_implementation(self) -> Any:
        cfg = dict(
            type="AIME24Dataset",
            path=self.path,
            name=self.subset if self.subset else "all",
            split=self.split
        )
        return await asyncio.to_thread(DATASET.build, cfg)

    def get_task_description(self) -> str:
        return "Please solve the following math contest problem. Put the final answer in \\boxed{}."

    async def _eval_logic(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """
        AIME 具体的提取与比对逻辑
        """
        extracted_pred = self._extract_answer(prediction)
        clean_gt = self._clean_number(ground_truth)
        
        if extracted_pred is None:
            return 0.0
        
        # AIME 只要清洗后的数字一致即为正确
        return 1.0 if extracted_pred == clean_gt else 0.0

    # --- Helper methods (保持不变) ---
    def _clean_number(self, text: str) -> str:
        if not text: return ""
        text = text.strip().replace(",", "").replace("$", "")
        try:
            return str(int(float(text)))
        except (ValueError, TypeError):
            return text

    def _extract_answer(self, prediction: str) -> Optional[str]:
        if not prediction: return None
        
        # 1. 优先匹配 \boxed{...}
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        matches = re.findall(boxed_pattern, prediction)
        if matches:
            return self._clean_number(matches[-1])
            
        # 2. 其次匹配 Answer: ...
        prompt_pattern = r"(?i)Answer:\s*(?:\$|\\\$)?\s*(-?[\d,]+)"
        matches = re.findall(prompt_pattern, prediction)
        if matches:
            return self._clean_number(matches[-1])
            
        # 3. 兜底提取最后一个数字 (慎用，但 AIME 这种纯数字题通常有效)
        numbers = re.findall(r"-?\d+", prediction)
        if numbers:
            return self._clean_number(numbers[-1])
            
        return None