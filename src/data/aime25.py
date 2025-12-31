import os
import json
import pandas as pd

from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(force=True)
class AIME25Dataset:
    def __init__(self, path, name, split):
        """
        Initialize AIME 2025 Dataset (Multi-folder support).
        
        Args:
            path: Base path to the dataset directory
            name: "all" or specific subset name ("AIME2025-I", "AIME2025-II")
            split: Dataset split ("test", "validation", etc.)
        """
        self.path = path
        self.name = name
        self.split = split

        # 1. 定义子集列表
        # 这里硬编码了两个文件夹名称
        all_subsets = ["AIME2025-I", "AIME2025-II"]

        # 2. 根据 name 参数决定加载哪些文件夹
        if name in all_subsets:
            target_subsets = [name]  # 只加载指定的一个文件夹
        else:
            target_subsets = all_subsets  # 默认加载所有文件夹 ("all")

        path = assemble_project_path(path)
        
        data_rows = []

        # 3. 遍历文件夹加载数据
        for subset_name in target_subsets:
            # 拼接文件路径：根目录 / split / 子文件夹名 / metadata.jsonl
            # 预期路径示例: /data/aime25/test/AIME2025-I/metadata.jsonl
            metadata_file = os.path.join(path, split, subset_name, "metadata.jsonl")
            
            # 容错：如果某个子集的文件夹不存在，打印警告但不报错，继续加载下一个
            if not os.path.exists(metadata_file):
                print(f"[Warning] Metadata file not found for subset {subset_name}: {metadata_file}")
                continue
            
            # 读取当前子集的数据
            with open(metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # 提取基础字段
                    raw_id = str(row.get("task_id", ""))
                    q_text = row.get("problem", "")
                    a_text = row.get("answer", "")
                    
                    if q_text:
                        # 构造新的唯一 task_id
                        # 为了防止两个文件夹里的 ID 都是从 1 开始导致冲突，
                        # 建议将文件夹名拼接到 ID 中，例如: "AIME2025-I_1"
                        if raw_id:
                            unique_id = f"{subset_name}_{raw_id}"
                        else:
                            # 如果原始数据没 ID，自动生成
                            unique_id = f"{subset_name}_{len(data_rows) + 1}"

                        data_row = {
                            "task_id": unique_id,
                            "question": q_text,
                            "true_answer": a_text,
                            "task": "AIME 2025",
                            "subset": subset_name, # 额外记录来源是哪个文件夹
                            "file_name": "" 
                        }
                        
                        # 答案格式化防御
                        if isinstance(data_row["true_answer"], (int, float)):
                            data_row["true_answer"] = str(int(data_row["true_answer"]))
                            
                        data_rows.append(data_row)
        
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