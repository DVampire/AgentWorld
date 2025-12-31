import re
from typing import Optional
from src.benchmark.types import Benchmark

class AIMEBenchmark(Benchmark):
    """
    AIME Benchmark Implementation
    针对特定 System Prompt ("Answer: $VALUE") 优化的评估函数
    """
    
    def _clean_number(self, text: str) -> str:
        """
        [Helper] 数字清洗
        去掉逗号，去掉空格，处理 float 转 int
        Example: "1,000" -> "1000", " 005 " -> "5", "5.0" -> "5"
        """
        if not text:
            return ""
        # 移除可能残留的 $ 符号 (以防正则漏掉) 和逗号
        text = text.strip().replace(",", "").replace("$", "")
        try:
            # 尝试转为 float 再转 int (处理 5.0)
            return str(int(float(text)))
        except (ValueError, TypeError):
            # 如果转换失败，返回原始清洗过的文本，后续比对时自然会失败
            return text

    def _extract_answer(self, prediction: str) -> Optional[str]:
        """
        [Helper] 从模型回复中提取答案
        
        策略优先级：
        1. System Prompt 格式: 'Answer: $VALUE' 或 'Answer: VALUE' (最高优先级)
        2. LaTeX 格式: \boxed{VALUE} (模型常用的训练习惯)
        3. GSM8k 风格: #### VALUE
        4. 兜底: 文本末尾的数字
        """
        if not prediction:
            return None

        # --- 策略 1: 严格匹配 System Prompt 要求的格式 ---
        # 目标匹配: "Answer: 123", "Answer: $123", "Answer: $123$"
        # 正则解释:
        # (?i)          -> 忽略大小写
        # Answer:\s* -> 匹配 "Answer:" 后跟任意空格
        # (?:\$|\\\$)?  -> 非捕获组：匹配可选的 "$" 或 "\$" (LaTeX转义)
        # (-?[\d,]+)    -> 捕获组：匹配数字，允许负号和逗号
        prompt_pattern = r"(?i)Answer:\s*(?:\$|\\\$)?\s*(-?[\d,]+)"
        
        # 使用 findall 获取所有匹配项，并取最后一个
        # 因为 System Prompt 要求 "The last line of your response..."
        matches = re.findall(prompt_pattern, prediction)
        if matches:
            return self._clean_number(matches[-1])

        # --- 策略 2: LaTeX \boxed{...} (常用兜底) ---
        # 即使 System Prompt 没要求，经过数学微调的模型依然倾向于输出这个
        boxed_pattern = r"\\boxed\{([^}]+)\}"
        matches = re.findall(boxed_pattern, prediction)
        if matches:
            return self._clean_number(matches[-1])

        # --- 策略 3: GSM8k 风格 (####) ---
        if "####" in prediction:
            return self._clean_number(prediction.split("####")[-1])

        # --- 策略 4: 极简兜底 (提取最后一个连续数字) ---
        # 仅当上述严格格式都失效时使用
        # 风险：可能会提取到 "step 5" 中的 5
        numbers = re.findall(r"-?\d+", prediction)
        if numbers:
            return self._clean_number(numbers[-1])

        return None

    async def eval(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """
        [Impl] AIME 评估逻辑
        
        Args:
            prediction: 模型生成的完整推理文本
            ground_truth: 纯数字答案 (如 "045" 或 "45")
        
        Returns:
            1.0 (Correct) or 0.0 (Incorrect)
        """
        # 1. 提取预测值
        extracted_pred = self._extract_answer(prediction)
        
        # 2. 清洗标准答案 (AIME 答案通常是 000-999 的整数)
        # 必须处理前导零，例如 GT 是 "007"，提取的是 "7"，这应该算对
        clean_gt = self._clean_number(ground_truth)
        
        # 3. 判空
        if extracted_pred is None:
            return 0.0
            
        # 4. 比较
        # 比较两个清洗后的字符串是否相等
        # Example: pred="7", gt="007" -> _clean_number("007")="7" -> Match
        is_correct = (extracted_pred == clean_gt)
        
        return 1.0 if is_correct else 0.0