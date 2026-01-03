# src/benchmark/manager.py

import asyncio
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

# 导入基类和具体实现
from .types import Benchmark
from .aime24 import AIME24Benchmark
from .aime25 import AIME25Benchmark
from .gpqa import GPQABenchmark
from .leetcode import LeetCodeBenchmark
# 注意：实际项目中建议使用动态注册或 importlib 避免循环依赖，
# 或者把 implementations 放在单独文件引用。

class BenchmarkFactory:
    """
    工厂类：根据名称创建对应的 Benchmark 实例
    """
    @staticmethod
    def create(name: str, **kwargs) -> Benchmark:
        name_lower = name.lower()
        
        # 映射表
        registry = {
            "aime24": AIME24Benchmark,
            "aime25": AIME25Benchmark,
            #"gsm8k": GSM8kBenchmark,
            "gpqa": GPQABenchmark,
            "leetcode": LeetCodeBenchmark
        }
        
        # 简单的别名匹配逻辑
        target_cls = None
        for key, cls in registry.items():
            if key in name_lower:
                target_cls = cls
                break
        
        if target_cls is None:
            raise ValueError(f"Unknown benchmark name: {name}")
            
        # 实例化
        return target_cls(benchmark_name=name, **kwargs)


class BenchmarkManager:
    """
    管理类：负责生命周期管理和批量执行
    """
    def __init__(self):
        self._current_benchmark: Optional[Benchmark] = None

    async def init_benchmark(self, name: str, path: str, split: str = "test", subset: str = None):
        """
        初始化并加载 Benchmark
        """
        # 1. 工厂创建实例
        self._current_benchmark = BenchmarkFactory.create(
            name=name, 
            path=path, 
            split=split, 
            subset=subset
        )
        
        # 2. 异步加载数据
        await self._current_benchmark.load()
        return self._current_benchmark

    def get_current_benchmark(self) -> Benchmark:
        if not self._current_benchmark:
            raise RuntimeError("No benchmark initialized.")
        return self._current_benchmark

    async def run_evaluation_batch(self, results: List[Dict[str, str]], concurrency: int = 10) -> float:
        """
        批量评估入口
        """
        if not self._current_benchmark:
            raise RuntimeError("Benchmark not initialized.")
            
        sem = asyncio.Semaphore(concurrency)

        async def _safe_eval(res):
            async with sem:
                t_id = str(res.get("task_id", ""))
                pred = res.get("prediction", "")
                
                if not t_id:
                    return 0.0
                
                try:
                    # 调用当前 Benchmark 实例的 eval_task 方法
                    # 实例内部会自动查找 GT 并调用特定的 _eval_logic
                    return await self._current_benchmark.eval_task(prediction=pred, task_id=t_id)
                except Exception as e:
                    print(f"[Error] Eval failed for {t_id}: {e}")
                    return 0.0

        tasks = [_safe_eval(res) for res in results]
        print(f"Starting batch evaluation for {len(tasks)} items...")
        scores = await asyncio.gather(*tasks)
        
        return sum(scores) / len(scores) if scores else 0.0