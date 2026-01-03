import os
import json
import pandas as pd

from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(force=True)
class AIME25Dataset:
    # 核心改动：使用 **kwargs 吸收掉所有不确定的参数（如 subset, name 等）
    def __init__(self, path, split, **kwargs):
        """
        Args:
            path: 数据集根目录 (例如 ./datasets/AIME25)
            split: 数据划分 (例如 test)
        """
        self.path = path
        self.split = split
        
        # 1. 转换路径并打印调试信息
        base_path = assemble_project_path(path)
        
        # 2. 自动定位 metadata.jsonl
        # 逻辑：在 {path}/{split}/ 下查找 metadata.jsonl
        metadata_file = os.path.join(base_path, split, "metadata.jsonl")
        
        # 🚨 调试：如果运行还是 0，请看控制台输出的这个路径在电脑里是否存在
        print(f"DEBUG: Dataset searching at -> {os.path.abspath(metadata_file)}")
        
        data_rows = []

        if not os.path.exists(metadata_file):
            print(f"❌ Error: File not found at {metadata_file}")
            self.data = pd.DataFrame()
            return

        # 3. 读取数据
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                q_text = row.get("problem") or row.get("question")
                a_text = row.get("answer")
                
                if q_text:
                    unique_id = str(row.get("task_id", f"aime_{len(data_rows)}"))

                    data_row = {
                        "task_id": unique_id,
                        "question": q_text,
                        "true_answer": str(a_text) if a_text is not None else "",
                        "task": "AIME 2025"
                    }
                    data_rows.append(data_row)
        
        self.data = pd.DataFrame(data_rows)
        print(f"✅ Successfully loaded {len(self.data)} tasks.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def get_task_description(self):
        return (
            "You will answer a challenging mathematics contest problem. Think step by step. "
            "State intermediate reasoning clearly. The last line of your response should be "
            "of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
        )