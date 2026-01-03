import os
import re
import asyncio
from typing import Optional, Dict, Any, List
from pydantic import PrivateAttr  # <--- 1. 新增导入
from .types import Benchmark
from src.registry import DATASET

# 假设 github_solver.py 在同级目录
from .code_submit import CodeSubmitter 

class LeetCodeBenchmark(Benchmark):
    """
    LeetCode 评测实现 (集成 Playwright 在线评测版)
    """
    
    # 2. 使用 PrivateAttr 声明内部状态
    # 这样 Pydantic 就不会报错 "object has no field"
    _id_to_record_map: Dict[str, Dict] = PrivateAttr(default_factory=dict)
    _submitter: Any = PrivateAttr(default=None)
    _submitter_started: bool = PrivateAttr(default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 3. 赋值给私有属性 (建议加下划线前缀)
        self._submitter = CodeSubmitter(headless=False)
        self._submitter_started = False

    async def _load_dataset_implementation(self) -> Any:
        cfg = dict(
            type="LeetCodeDataset",
            path=self.path,
            split=self.split,
            name=self.subset if self.subset else None
        )
        dataset = await asyncio.to_thread(DATASET.build, cfg)
        
        # 使用私有属性
        self._id_to_record_map = {}
        for record in dataset.data.to_dict(orient="records"):
            tid = str(record.get("id") or record.get("task_id", "0"))
            self._id_to_record_map[tid] = record
            
        print(f"[{self.benchmark_name}] Index built: {len(self._id_to_record_map)} records mapped.")
        return dataset

    def step(self) -> Optional[Dict[str, Any]]:
        task = super().step()
        if task is None:
            return None
        
        if not task.get("task_id") and "id" in task:
            task["task_id"] = str(task["id"])
            
        templates = task.get("code_template", {})
        py_template = templates.get("python3") or templates.get("python", "")
        
        if py_template:
            task["input"] = (
                f"{task['input']}\n\n"
                f"Please write the solution based on this template:\n"
                f"```python\n{py_template}\n```"
            )

        return task

    def get_task_description(self) -> str:
        return (
            "You will solve a LeetCode algorithmic problem. "
            "The last part of your response must be the Python solution code, "
            "strictly enclosed between the standard LeetCode markers as follows:\n\n"
            "# @lc code=start\n"
            "class Solution:\n"
            "    # ... implementation ...\n"
            "# @lc code=end\n\n"
            "Do not wrap this block in markdown code ticks."
        )

    async def _eval_logic(self, prediction: str, ground_truth: str, task_id: str, **kwargs) -> float:
        """
        执行在线评测逻辑
        """
        code_body = self._extract_inner_code(prediction)
        if not code_body:
            print(f"[{task_id}] No code extracted.")
            return 0.0

        name = kwargs.get("name")
        # 使用私有属性 _id_to_record_map
        if not name and task_id in self._id_to_record_map:
            name = self._id_to_record_map[task_id].get("name")
        if not name:
            name = "Unknown Problem"

        full_content = (
            f"#\n"
            f"# @lc app=leetcode id={task_id} lang=python3\n"
            f"#\n"
            f"# [{task_id}] {name}\n"
            f"#\n\n"
            f"# @lc code=start\n"
            f"{code_body}\n"
            f"# @lc code=end\n"
        )

        # 4. 使用私有属性 _submitter 和 _submitter_started
        if not self._submitter_started:
            print("[LeetCodeBenchmark] Starting CodeSubmitter browser...")
            try:
                # 注意这里调用的是 self._submitter
                await asyncio.to_thread(self._submitter.start)
                self._submitter_started = True
            except Exception as e:
                print(f"[{task_id}] Failed to start submitter: {e}")
                return 0.0

        try:
            print(f"[{task_id}] Submitting code...")
            # 注意这里调用的是 self._submitter
            result = await asyncio.to_thread(self._submitter.submit_code, full_content)
            
            score = self._parse_result_score(result)
            print(f"[{task_id}] Eval Result: {result['status']} | Score: {score:.2f}")
            return score

        except Exception as e:
            print(f"[{task_id}] Submission error: {e}")
            return 0.0

    def _parse_result_score(self, result: Dict[str, Any]) -> float:
        status = result.get("status", "")
        details = result.get("details", [])

        if status == "Accepted":
            return 1.0

        if status == "Compile Error":
            return 0.0

        for line in details:
            match = re.search(r"(\d+)/(\d+)\s+cases\s+passed", line)
            if match:
                passed = int(match.group(1))
                total = int(match.group(2))
                
                if total > 0:
                    return passed / total
        
        return 0.0

    def _extract_inner_code(self, text: str) -> Optional[str]:
        if not text: return None
        pattern = r"# @lc code=start\s+(.*?)# @lc code=end"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
        
    def __del__(self):
        # 5. 使用私有属性清理
        if hasattr(self, '_submitter') and self._submitter_started:
            try:
                self._submitter.close()
            except:
                pass