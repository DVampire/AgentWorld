import asyncio
import inspect
import pandas as pd
from typing import Optional, Callable, Any, List, Dict, Union, Awaitable

from pydantic import Field, PrivateAttr
from .types import Benchmark
from src.registry import DATASET

class BenchmarkManager(Benchmark):
    """
    [防泄露版] 异步基准测试管理器
    特点：对外隐藏答案，评估时通过 task_id 内部查找答案
    """
    # 评估函数: 输入变为 (prediction, ground_truth) -> float
    evaluate_fn: Optional[Union[
        Callable[[str, str], float], 
        Callable[[str, str], Awaitable[float]]
    ]] = Field(default=None, description="底层评测函数")
    
    # 私有属性
    _dataset_instance: Any = PrivateAttr(default=None)
    # 关键：私有的答案映射表 {task_id: true_answer}
    _ground_truth_map: Dict[str, str] = PrivateAttr(default_factory=dict)

    def _get_dataset_type(self) -> str:
        mapping = {
            "esg": "ESGDataset",
            "aime24": "AIME24Dataset",
            "aime25": "AIME25Dataset",
            "gpqa": "GPQADataset",
            "gsm8k": "GSM8kDataset",
            "leetcode": "LeetCodeDataset"
        }
        b_name = self.benchmark_name.lower()
        if b_name not in mapping:
            reverse_values = {v.lower(): v for v in mapping.values()}
            if b_name in reverse_values:
                return reverse_values[b_name]
            raise ValueError(f"未知的 benchmark 名称: {b_name}")
        return mapping[b_name]

    async def load(self):
        """
        [Async] 异步加载数据集，并构建私有答案索引
        """
        dataset_type = self._get_dataset_type()
        cfg = dict(
            type=dataset_type,
            path=self.path,
            name=self.subset if self.subset else "all",
            split=self.split
        )
        if self.benchmark_name.lower() == "leetcode":
            cfg['lang'] = 'python3'

        print(f"[BenchmarkManager] Async Loading: {dataset_type} ...")
        
        try:
            # 1. 完整加载数据（包含答案）
            self._dataset_instance = await asyncio.to_thread(DATASET.build, cfg)
            
            # 2. 构建私有答案映射表 (内存操作，速度极快，无需 async)
            # 假设所有 Dataset 都有 task_id 和 true_answer 字段
            full_df = self._dataset_instance.data
            
            if "task_id" in full_df.columns and "true_answer" in full_df.columns:
                # 将 task_id 和 answer 转为字典: {'id_1': 'ans_1', 'id_2': 'ans_2'}
                self._ground_truth_map = dict(zip(
                    full_df["task_id"].astype(str), 
                    full_df["true_answer"].astype(str)
                ))
            else:
                print(f"[Warning] 数据集缺少 task_id 或 true_answer 字段，无法构建防泄露索引。")

            print(f"[BenchmarkManager] Loaded. Size: {len(self._dataset_instance)}")
            print(f"[Security] Answers have been hidden and cached internally.")
            
        except Exception as e:
            raise RuntimeError(f"异步加载失败: {e}")

    def get_data(self) -> pd.DataFrame:
        """
        [Safe] 获取脱敏后的数据
        移除 'true_answer', 'origin_answer', 'reasoning' 等敏感列
        """
        if self._dataset_instance is None:
            raise RuntimeError("数据集尚未加载，请先调用 await manager.load()")
        
        # 获取原始数据的深拷贝
        df = self._dataset_instance.data.copy()
        
        # 定义需要隐藏的敏感字段
        sensitive_cols = ["true_answer", "origin_answer", "reasoning", "solution", "answer"]
        
        # 仅删除存在的列，忽略不存在的错误
        safe_df = df.drop(columns=sensitive_cols, errors="ignore")
        
        return safe_df

    def get_instance(self):
        """
        [Warning] 此方法返回底层实例，依然包含答案。
        如果非常严格，建议删除此方法或警告。
        """
        return self._dataset_instance
    
    
    async def reset(self, benchmark_name: str, split: Optional[str] = None):
        
        if split is None:
            split = "test"

        benchmark = self._benchmarks[benchmark_name]
        await benchmark.reset(split=split)
        
    async def eval(self, prediction: str, task_id: str, **kwargs) -> float:
        """
        [Async] 执行评估
        用户只需传入 prediction 和 task_id，无需（也无法）传入 ground_truth
        """
        if not self.evaluate_fn:
            raise NotImplementedError("未提供 evaluate_fn")

        # 1. 内部查找 Ground Truth
        ground_truth = self._ground_truth_map.get(str(task_id))
        
        if ground_truth is None:
            print(f"[Error] Task ID '{task_id}' not found in ground truth map. Skipping.")
            return 0.0

        # 2. 调用具体的评测逻辑
        if inspect.iscoroutinefunction(self.evaluate_fn):
            return await self.evaluate_fn(prediction, ground_truth, **kwargs)
        else:
            return self.evaluate_fn(prediction, ground_truth, **kwargs)
    
    # ================= 新增的核心函数 =================
    def get_task_description(self) -> str:
        """
        获取当前数据集的任务描述 (System Prompt)。
        调用底层 Dataset 实例的 get_task_description 方法。
        """
        # 1. 检查是否加载
        if self._dataset_instance is None:
            # 这是一个友好的 fallback，或者你可以选择 raise RuntimeError
            return f"You are a helpful assistant solving tasks for {self.benchmark_name}."

        # 2. 尝试调用底层 Dataset 的方法
        if hasattr(self._dataset_instance, "get_task_description"):
            try:
                # 调用底层方法
                desc = self._dataset_instance.get_task_description()
                if desc and isinstance(desc, str):
                    return desc
            except Exception as e:
                print(f"[Warning] 调用数据集 get_task_description 失败: {e}")
        
        # 3. 如果底层没有定义该方法，返回默认兜底描述
        default_prompts = {
            "gsm8k": "You are a helpful assistant. Solve the following math problem step by step.",
            "gpqa": "You are an expert scientist. Answer the multiple choice question.",
            "aime24": "Please solve the following math contest problem. Put the final answer in \\boxed{}.",
            "leetcode": "You are an expert programmer. Write code to solve the algorithmic problem."
        }
        
        # 模糊匹配默认 Prompt
        for key, prompt in default_prompts.items():
            if key in self.benchmark_name.lower():
                return prompt
                
        return "You are a helpful AI assistant."
    # =================================================

    async def run_evaluation_batch(self, results: List[Dict[str, str]], concurrency: int = 10) -> float:
        """
        [Async] 批量并发评估
        Args:
            results: 包含 [{"task_id": "...", "prediction": "..."}, ...] 的列表
        """
        if not self.evaluate_fn:
            print("[Warning] No evaluate_fn provided.")
            return 0.0

        sem = asyncio.Semaphore(concurrency)

        async def _safe_eval(res):
            async with sem:
                # 获取 task_id 和 prediction
                t_id = str(res.get("task_id", ""))
                pred = res.get("prediction", "")
                
                if not t_id:
                    return 0.0
                
                try:
                    # 调用修改后的 eval，内部自动 lookup 答案
                    return await self.eval(prediction=pred, task_id=t_id)
                except Exception as e:
                    print(f"[Error] Eval failed for {t_id}: {e}")
                    return 0.0

        tasks = [_safe_eval(res) for res in results]
        print(f"Starting secure batch evaluation for {len(tasks)} items...")
        scores = await asyncio.gather(*tasks)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return avg_score
    
    
benchmark_manager = BenchmarkManager()