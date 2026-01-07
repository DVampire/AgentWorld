import os
import json
import re
import asyncio
from typing import Optional, Dict, Any, Set
from pydantic import PrivateAttr

from src.benchmark.types import Benchmark
from src.registry import BENCHMARK
from src.benchmark.code_submit import CodeSubmitter 

@BENCHMARK.register_module(force=True)
class LeetCodeBenchmark(Benchmark):
    """
    LeetCode Benchmark with Resume Capability.
    Automatically filters out tasks present in 'tmp/answer.jsonl'.
    """
    name: str = "leetcode"
    path: str = "datasets/leetcode"
    
    _id_to_record_map: Dict[str, Dict] = PrivateAttr(default_factory=dict)
    _submitter: Any = PrivateAttr(default=None)
    _submitter_started: bool = PrivateAttr(default=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submitter = CodeSubmitter(headless=False)
        self._submitter_started = False
        os.makedirs("tmp", exist_ok=True)
        
        # ✅ 新增：初始化完成后，自动过滤掉已经跑完的任务
        self._filter_finished_tasks()

    def _filter_finished_tasks(self):
        """检查 answer.jsonl，过滤掉已完成的 task_id"""
        jsonl_path = os.path.join("tmp", "answer.jsonl")
        if not os.path.exists(jsonl_path):
            return

        finished_ids: Set[str] = set()
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        record = json.loads(line)
                        # 确保 task_id 是字符串格式，方便比对
                        if "task_id" in record:
                            finished_ids.add(str(record["task_id"]))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"[{self.name}] ⚠️ Error reading existing logs: {e}")

        if not finished_ids:
            return

        # 过滤 self._data_records
        original_count = len(self._data_records)
        self._data_records = [
            r for r in self._data_records 
            if str(r.get("id") or r.get("task_id")) not in finished_ids
        ]
        new_count = len(self._data_records)
        skipped_count = original_count - new_count

        if skipped_count > 0:
            print(f"| [{self.name}] ⏭️ Resuming: Skipped {skipped_count} tasks (Already processed). Remaining: {new_count}")
        else:
            print(f"| [{self.name}] 🔄 No tasks skipped. Starting fresh.")

    def _instantiate_dataset(self) -> Any:
        try:
            from src.data.leetcode import LeetCodeDataset
            dataset = LeetCodeDataset(
                path=self.path,
                split=self.split,
                name=self.subset if self.subset else None
            )
            self._id_to_record_map = {}
            if hasattr(dataset, 'data'):
                for record in dataset.data.to_dict(orient="records"):
                    tid = str(record.get("id") or record.get("task_id", "0"))
                    self._id_to_record_map[tid] = record
            print(f"[{self.name}] Index built: {len(self._id_to_record_map)} records mapped.")
            return dataset
        except ImportError:
            return None

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
            "You are a strictly formatted LeetCode coding assistant. "
            "Your only output must be the Python solution code, strictly enclosed between specific markers. "
            "Do NOT wrap the code in markdown blocks (like ```python). "
            "Here is an example of the required output:\n\n"
            "--- Example ---\n"
            "# @lc code=start\n"
            "class Solution:\n"
            "    def sum(self, num1: int, num2: int) -> int:\n"
            "        return num1 + num2\n"
            "# @lc code=end\n"
            "--- End Example ---\n\n"
            "Now, solve the actual problem provided. Output strictly the code block as shown above."
        )

    async def _eval_logic(self, prediction: str, ground_truth: str, task_id: str, **kwargs) -> float:
        code_body = self._extract_inner_code(prediction)
        if not code_body:
            print(f"[{task_id}] ❌ No code extracted.")
            return 0.0

        name = "Unknown_Problem"
        if task_id in self._id_to_record_map:
            raw_name = self._id_to_record_map[task_id].get("name") or "Unknown"
            name = re.sub(r'[\\/*?:"<>| ]', '_', str(raw_name))

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

        if not self._submitter_started:
            print("[LeetCodeBenchmark] Starting CodeSubmitter browser...")
            try:
                await asyncio.to_thread(self._submitter.start)
                self._submitter_started = True
            except Exception as e:
                print(f"[{task_id}] ❌ Failed to start submitter: {e}")
                return 0.0

        try:
            print(f"[{task_id}] 🚀 Submitting code...")
            result = await asyncio.to_thread(self._submitter.submit_code, full_content)
            
            self._save_artifacts(task_id, name, full_content, result)

            score = self._parse_result_score(result)
            print(f"[{task_id}] Result: {result.get('status')} | Score: {score:.2f}")
            return score

        except Exception as e:
            print(f"[{task_id}] ❌ Submission error: {e}")
            import traceback
            traceback.print_exc()
            return 0.0

    def _save_artifacts(self, task_id: str, task_name: str, code_content: str, result: Dict[str, Any]):
        try:
            code_filename = os.path.join("tmp", f"{task_id}_{task_name}.py")
            with open(code_filename, "w", encoding="utf-8") as f:
                f.write(code_content)
            
            status = result.get("status", "Unknown")
            details = result.get("details", [])
            
            passed_cases = 0
            total_cases = 0
            runtime = -1.0
            memory_usage = -1.0
            runtime_beats = -1.0
            memory_beats = -1.0
            
            for line in details:
                if not isinstance(line, str): continue
                
                case_match = re.search(r"(\d+)/(\d+)\s+cases\s+passed", line)
                if case_match:
                    passed_cases = int(case_match.group(1))
                    total_cases = int(case_match.group(2))
                    
                    rt_match = re.search(r"\((\d+(?:\.\d+)?)\s*ms\)", line)
                    if rt_match:
                        runtime = float(rt_match.group(1))

                mem_match = re.search(r"\(([\d\.]+)\s*MB\)", line)
                if mem_match:
                    memory_usage = float(mem_match.group(1))
                    
                rt_beats_match = re.search(r"runtime beats\s+(\d+(?:\.\d+)?)\s*%", line)
                if rt_beats_match:
                    runtime_beats = float(rt_beats_match.group(1))
                
                mem_beats_match = re.search(r"memory usage beats\s+(\d+(?:\.\d+)?)\s*%", line)
                if mem_beats_match:
                    memory_beats = float(mem_beats_match.group(1))
            
            log_entry = {
                "task_id": task_id,
                "status": status,
                "total_cases": total_cases,
                "passed_cases": passed_cases,
                "runtime": runtime,
                "memory_usage": memory_usage,
                "runtime_beats": runtime_beats,
                "memory_beats": memory_beats
            }
            
            jsonl_path = os.path.join("tmp", "answer.jsonl")
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                
            print(f"[{task_id}] 💾 Stats saved: Runtime={runtime}, Mem={memory_usage}")

        except Exception as e:
            print(f"[{task_id}] ⚠️ Failed to save artifacts: {e}")

    def _parse_result_score(self, result: Dict[str, Any]) -> float:
        status = result.get("status", "")
        details = result.get("details", [])

        if status == "Accepted":
            return 1.0
        if status == "Compile Error":
            return 0.0
        
        for line in details:
            if isinstance(line, str):
                match = re.search(r"(\d+)/(\d+)\s+cases\s+passed", line)
                if match:
                    passed = int(match.group(1))
                    total = int(match.group(2))
                    return passed / total if total > 0 else 0.0
        return 0.0

    def _extract_inner_code(self, text: str) -> Optional[str]:
        if not text: return None
        pattern = r"# @lc code=start\s+(.*?)# @lc code=end"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
        
    def __del__(self):
        if hasattr(self, '_submitter') and self._submitter_started:
            try:
                if hasattr(self._submitter, 'close'):
                    self._submitter.close()
            except Exception:
                pass