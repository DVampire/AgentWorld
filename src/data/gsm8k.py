import os
import json
import pandas as pd

from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(force=True)
class GSM8kDataset:
    def __init__(self, path, name, split):
        """
        Initialize GSM8k Dataset (Supports 'main' and 'socratic' subsets).
        
        Args:
            path: Base path to the dataset directory
            name: "all", "main", or "socratic"
            split: Dataset split ("test", "train", "validation")
        """
        self.path = path
        self.name = name
        self.split = split

        # 1. 定义子集列表
        all_subsets = ["main", "socratic"]

        # 2. 确定要加载的子集
        if name == "all":
            target_subsets = all_subsets
        elif name in all_subsets:
            target_subsets = [name]
        else:
            # 容错：如果输入未知的名称，默认加载 main，或者你可以改为报错
            print(f"[Warning] Unknown subset '{name}'. Defaulting to 'main'.")
            target_subsets = ["main"]

        path = assemble_project_path(path)
        data_rows = []

        # 3. 遍历子集加载
        for subset_name in target_subsets:
            # 预期路径: /data/gsm8k/test/main/metadata.jsonl
            metadata_file = os.path.join(path, split, subset_name, "metadata.jsonl")
            
            if not os.path.exists(metadata_file):
                print(f"[Warning] Metadata file not found for {subset_name}: {metadata_file}")
                continue
            
            with open(metadata_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # --- GSM8k 特有的答案解析逻辑 ---
                    # 原始 answer 格式示例: "Janet sells ... <<9*2=18>>18.\n#### 18"
                    raw_answer = row.get("answer", "")
                    
                    reasoning_content = ""
                    final_answer = ""
                    
                    if "####" in raw_answer:
                        # 依据 '####' 符号进行分割
                        parts = raw_answer.split("####")
                        # 前半部分是推理过程 (CoT)
                        reasoning_content = parts[0].strip()
                        # 后半部分是最终数值答案 (Gold Answer)
                        # strip() 去除可能存在的换行符或空格
                        final_answer = parts[-1].strip()
                    else:
                        # 异常处理：如果没有 ####，则整个作为答案
                        final_answer = raw_answer.strip()

                    # --- 构造数据行 ---
                    # 确保 task_id 全局唯一
                    raw_id = str(row.get("task_id", ""))
                    if raw_id:
                        unique_id = f"{subset_name}_{raw_id}"
                    else:
                        unique_id = f"{subset_name}_{len(data_rows)+1}"

                    data_row = {
                        "task_id": unique_id,
                        
                        "question": row.get("question", ""),
                        
                        # true_answer 只存放最终数值 (例如 "18")
                        # 这是为了方便后续用 exact match 或数值比较进行评测
                        "true_answer": final_answer,
                        
                        # 我们把完整的推理过程也存下来，万一通过 prompt learning 需要用到
                        "reasoning": reasoning_content,
                        
                        "task": "GSM8k",
                        "subset": subset_name, # 标记是 main 还是 socratic
                        "file_name": ""
                    }
                    
                    # 简单防御：去除答案中的逗号 (例如 "1,000" -> "1000")
                    # 这样能提高数值匹配的准确率
                    if isinstance(data_row["true_answer"], str):
                        data_row["true_answer"] = data_row["true_answer"].replace(",", "")

                    data_rows.append(data_row)
        
        self.data = pd.DataFrame(data_rows)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def get_task_description(self):
        return "You will answer a mathemetical reasoning question. Think step by step. The last line of your response should be of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
