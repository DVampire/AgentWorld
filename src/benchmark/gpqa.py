# src/benchmark/implementations.py

import re
import asyncio
from typing import Optional, Any
from .types import Benchmark
from src.registry import DATASET

class GPQABenchmark(Benchmark):
    """
    GPQA (Google-Proof Q&A) 具体实现
    类型：多项选择题 (Multiple Choice)
    目标：评估模型在复杂科学问答中的准确率
    """
    
    async def _load_dataset_implementation(self) -> Any:
        cfg = dict(
            type="GPQADataset",  # 假设注册表中叫这个
            path=self.path,
            # GPQA 有 gpqa_diamond, gpqa_main, gpqa_extended 等子集
            # 如果没指定 subset，默认使用 gpqa_diamond
            name=self.subset if self.subset else "gpqa_diamond", 
            split=self.split
        )
        return await asyncio.to_thread(DATASET.build, cfg)

    def get_task_description(self) -> str:
        """
        针对选择题的 System Prompt
        """
        return (
            "You are an expert scientist. Answer the following multiple-choice question. "
            "Select the correct answer from the given options. "
            "End your response with the format 'Answer: <letter>' (e.g., Answer: A)."
        )

    async def _eval_logic(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """
        GPQA 具体的提取与比对逻辑
        Args:
            prediction: 模型输出的完整文本
            ground_truth: 正确选项的字母，如 "A", "B", "C", "D"
        """
        extracted_pred = self._extract_answer(prediction)
        clean_gt = self._clean_choice(ground_truth)
        
        if extracted_pred is None:
            return 0.0
        
        # 只要提取出的字母与 GT 字母一致即为正确
        return 1.0 if extracted_pred == clean_gt else 0.0

    # --- Helper methods ---

    def _clean_choice(self, text: str) -> str:
        """
        清洗选项字母
        Examples: "(A)" -> "A", " A. " -> "A", "c" -> "C"
        """
        if not text:
            return ""
        # 移除空格、句号、括号，并转大写
        text = text.strip().upper()
        text = text.replace("(", "").replace(")", "").replace(".", "")
        return text

    def _extract_answer(self, prediction: str) -> Optional[str]:
        """
        从模型回复中提取 A/B/C/D 选项
        
        策略优先级：
        1. 严格格式: "Answer: A"
        2. LaTeX/括号格式: \boxed{A}, (A)
        3. 句末模式: "The answer is A"
        """
        if not prediction:
            return None

        # 预处理：只保留最后一部分文本可能会更安全，但这里先分析全文
        # 很多模型会输出 'Answer: (A)' 或 'Answer: A'
        
        # --- 策略 1: 严格匹配 "Answer: X" ---
        # 匹配 "Answer: A", "Answer: (B)", "answer: C."
        # (?i) 忽略大小写
        # [A-D] 限制只匹配 A, B, C, D (如果题目可能有E/F需扩展)
        prompt_pattern = r"(?i)Answer:\s*\(?([A-D])\)?"
        matches = re.findall(prompt_pattern, prediction)
        if matches:
            return self._clean_choice(matches[-1])

        # --- 策略 2: LaTeX \boxed{X} ---
        # 有些经过数学微调的模型习惯给字母加框
        boxed_pattern = r"\\boxed\{\s*([A-D])\s*\}"
        matches = re.findall(boxed_pattern, prediction)
        if matches:
            return self._clean_choice(matches[-1])

        # --- 策略 3: 匹配括号选项 (A) ---
        # 如果模型输出 "Therefore, the correct option is (C)."
        paren_pattern = r"\s\(([A-D])\)" 
        matches = re.findall(paren_pattern, prediction)
        if matches:
            return self._clean_choice(matches[-1])

        # --- 策略 4: 极简兜底 (仅当非常确信时使用) ---
        # 提取最后出现的单独字母 A-D
        # 风险：如果不严格限制，可能会提取到单词里的字母。
        # 这里限制为：行首或空格后 + 字母 + 标点或行尾
        loose_pattern = r"(?:^|\s)([A-D])(?:\.|,|$)"
        matches = re.findall(loose_pattern, prediction)
        if matches:
            return self._clean_choice(matches[-1])

        return None