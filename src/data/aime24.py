import os
import json
import pandas as pd

from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(force=True)
class AIME24Dataset:
    def __init__(self, path, name, split):
        """
        Initialize AIME 2024 Dataset.
        
        Args:
            path: Base path to the dataset directory
            name: Dataset name (Not used for filtering, kept for compatibility)
            split: Dataset split ("test", "validation", etc.)
        """
        self.path = path
        self.name = name
        self.split = split

        # 1. 路径处理
        path = assemble_project_path(path)
        
        # 2. 定位 metadata.jsonl 文件
        metadata_file = os.path.join(path, split, "metadata.jsonl")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # 3. 读取并清洗数据
        data_rows = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue # 跳过损坏的行

                # --- 针对你的数据格式 {"task_id": 60, "problem": "...", "answer": "204"} ---
                
                # 提取字段
                # 注意：原始 task_id 是整数 60，建议转为字符串以保持通用性
                t_id = str(row.get("task_id", ""))
                q_text = row.get("problem", "")
                a_text = row.get("answer", "")
                
                # 简单校验：只要有题目文本，就视为有效数据
                if q_text:
                    data_row = {
                        "task_id": t_id,          # 对应 json 中的 task_id
                        "question": q_text,       # 将 problem 映射为框架通用的 question
                        "true_answer": a_text,    # 将 answer 映射为框架通用的 true_answer
                        "task": "AIME 2024",      # 固定标签，方便后续识别来源
                        "file_name": ""           # AIME 题目通常是纯文本，无附件
                    }
                    
                    # 答案防御性处理：万一 json 里 answer 是数字类型的 204 而不是字符串 "204"
                    if isinstance(data_row["true_answer"], (int, float)):
                        data_row["true_answer"] = str(int(data_row["true_answer"]))
                        
                    data_rows.append(data_row)
        
        # 4. 转为 DataFrame
        self.data = pd.DataFrame(data_rows)
    
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