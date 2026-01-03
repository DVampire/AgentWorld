# src/benchmark/base.py

import pandas as pd
import asyncio
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

class Benchmark(BaseModel):
    """
    基准测试抽象基类 (Stateful)
    增加了进度管理、步进执行和结果统计功能
    """
    benchmark_name: str = Field(..., description="基准测试名称")
    split: str = Field(default="test", description="数据集划分")
    subset: Optional[str] = Field(default=None, description="子集名称")
    path: str = Field(..., description="数据集路径")
    
    # --- 状态管理私有属性 ---
    _dataset_instance: Any = PrivateAttr(default=None)
    _ground_truth_map: Dict[str, str] = PrivateAttr(default_factory=dict)
    
    # 迭代器状态
    _data_records: List[Dict] = PrivateAttr(default_factory=list) # 缓存数据列表，方便索引
    _current_index: int = PrivateAttr(default=0)
    
    # 结果统计
    _results: List[Dict] = PrivateAttr(default_factory=list) # 存储每一条的评测详情

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def load(self):
        """
        加载数据并初始化迭代器状态
        """
        print(f"[{self.benchmark_name}] Async Loading...")
        self._dataset_instance = await self._load_dataset_implementation()
        
        # 构建防泄露索引
        full_df = self._dataset_instance.data
        if "task_id" in full_df.columns and "true_answer" in full_df.columns:
            self._ground_truth_map = dict(zip(
                full_df["task_id"].astype(str), 
                full_df["true_answer"].astype(str)
            ))
        
        # 缓存数据用于 step() 迭代
        # 我们这里把 DataFrame 转为 Dict List，方便通过 index 访问
        self._data_records = full_df.to_dict(orient="records")
        
        # 初始化状态
        self.reset()
        
        print(f"[{self.benchmark_name}] Loaded. Size: {len(self._data_records)}")

    async def _load_dataset_implementation(self) -> Any:
        raise NotImplementedError

    def get_task_description(self) -> str:
        raise NotImplementedError

    # ================= 状态与控制函数 =================

    def reset(self):
        """
        重置评测进度和统计信息
        """
        self._current_index = 0
        self._results = []
        print(f"[{self.benchmark_name}] Progress reset. Ready to start.")

    def step(self) -> Optional[Dict[str, Any]]:
        """
        [Iterator] 获取下一个待测任务
        
        Returns:
            Dict: 包含任务数据的字典 (task_id, question, system_prompt 等)
            None: 如果已经遍历完所有数据
        """
        if self._current_index >= len(self._data_records):
            return None
        
        # 获取当前记录
        record = self._data_records[self._current_index]
        self._current_index += 1
        
        # 构造返回给 Runner 的任务对象
        # 自动附带 system_prompt 方便调用
        task = {
            "task_id": str(record.get("task_id") or record.get("id","")),
            "input": record.get("question") or record.get("prompt") or "", # 兼容不同字段名
            "system_prompt": self.get_task_description(),
            # 保留原数据中可能需要的额外字段 (如 images)
            **{k: v for k, v in record.items() if k not in ["true_answer", "answer"]}
        }
        return task

    @property
    def progress(self) -> Tuple[int, int]:
        """返回 (当前索引, 总数)"""
        return self._current_index, len(self._data_records)

    # ================= 评估与统计函数 =================

    async def eval_task(self, prediction: str, task_id: str, **kwargs) -> float:
        """
        对外暴露的单条评估接口：
        1. 查找答案
        2. 计算分数
        3. **记录结果到历史列表**
        """
        ground_truth = self._ground_truth_map.get(str(task_id))
        
        if ground_truth is None:
            print(f"[Error] Task ID '{task_id}' not found.")
            score = 0.0
            ground_truth = "N/A" # 标记缺失
        else:
            score = await self._eval_logic(prediction, ground_truth, task_id=str(task_id), **kwargs)
            
        # --- 记录状态 ---
        result_entry = {
            "task_id": task_id,
            "prediction": prediction,
            "ground_truth": ground_truth, # 注意：结果里包含 GT，如果需要导出给用户看需脱敏
            "score": score,
            "metadata": kwargs
        }
        self._results.append(result_entry)
        
        return score

    async def _eval_logic(self, prediction: str, ground_truth: str, **kwargs) -> float:
        raise NotImplementedError

    def get_stats(self) -> Dict[str, Any]:
        """
        计算当前的整体统计数据
        """
        total_attempted = len(self._results)
        if total_attempted == 0:
            return {"accuracy": 0.0, "total": 0, "correct": 0}
        
        correct_count = sum(1 for r in self._results if r["score"] >= 1.0) # 假设 1.0 是全对
        accuracy = correct_count / total_attempted
        
        return {
            "accuracy": accuracy,
            "total_attempted": total_attempted,
            "correct_count": correct_count,
            "total_dataset_size": len(self._data_records),
            "progress_percent": f"{(total_attempted / len(self._data_records))*100:.1f}%"
        }

    def get_results(self) -> List[Dict]:
        """获取详细结果列表"""
        return self._results