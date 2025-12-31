# src/benchmark/base.py

import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

class Benchmark(BaseModel):
    """
    基准测试基类 (Async Base Class)
    """
    benchmark_name: str = Field(..., description="基准测试名称，如 gsm8k, gpqa")
    split: str = Field(default="test", description="数据集划分")
    subset: Optional[str] = Field(default=None, description="子集名称")
    path: str = Field(..., description="数据集根目录路径")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def load(self):
        """[Async Abstract] 异步加载数据接口"""
        raise NotImplementedError

    def get_data(self) -> pd.DataFrame:
        """[Sync] 获取数据"""
        raise NotImplementedError

    def get_task_description(self) -> str:
        """
        [Sync] 获取任务描述/系统提示词
        通常用于构建 System Prompt
        """
        raise NotImplementedError

    async def eval(self, prediction: str, ground_truth: str, **kwargs) -> float:
        """[Async Abstract] 异步单条评估接口"""
        raise NotImplementedError