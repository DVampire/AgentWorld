import os
import json
import pandas as pd

from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(force=True)
class LeetCodeDataset:
    def __init__(self, path, name, split, lang="python3"):
        """
        Initialize LeetCode Dataset.
        
        Args:
            path: Base path to the dataset directory
            name: Dataset mode (e.g., "all", or "easy", "hard" if folders are split)
            split: Dataset split ("test", "validation", "train")
            lang: The target programming language for the code template (default: "python3")
                  Supported: "cpp", "java", "python3", "golang", "rust", etc.
        """
        self.path = path
        self.name = name
        self.split = split
        self.lang = lang

        # 1. 路径标准化
        #path = assemble_project_path(path)
        
        # 2. 加载元数据文件
        # 假设路径结构: /data/leetcode/test/metadata.jsonl
        metadata_file = os.path.join(path, split, "metadata.jsonl")
        
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        data_rows = []
        
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # --- LeetCode 特有的加载逻辑 ---

                # 1. 读取题目描述 (Markdown 内容)
                # row['file'] 示例: "./question/1.two_sum.md"
                # 我们需要将其解析为绝对路径。通常它是相对于 metadata.jsonl 所在目录的
                rel_file_path = row.get("file", "")
                question_content = ""
                
                if rel_file_path:
                    # 移除可能的 "./" 前缀，拼接完整路径
                    # 假设 question 文件夹和 metadata.jsonl 在同一级 split 目录下
                    clean_rel_path = rel_file_path.replace("./", "") 
                    abs_file_path = os.path.join(path, split, clean_rel_path)
                    
                    if os.path.exists(abs_file_path):
                        with open(abs_file_path, "r", encoding="utf-8") as qf:
                            question_content = qf.read()
                    else:
                        print(f"[Warning] Question file not found: {abs_file_path}")
                        # 如果找不到描述文件，这道题无法做，跳过
                        continue
                
                # 2. 获取指定语言的代码模板
                templates = row.get("code_template", {})
                # 获取用户指定的语言，如果找不到，尝试 fallback 到 python3 或第一个可用的
                code_stub = templates.get(self.lang, "")
                if not code_stub and self.lang == "python3":
                    # 尝试找 python (老版本字段)
                    code_stub = templates.get("python", "")
                
                # 3. 构建 Prompt
                # 典型的代码生成 Prompt: 描述 + 代码起手式
                full_prompt = f"Problem Description:\n{question_content}\n\n" \
                              f"Please write a solution in {self.lang}:\n```\n{code_stub}\n```"

                # 4. 构造数据行
                data_row = {
                    "task_id": str(row.get("id", "")), # 转字符串
                    
                    "name": row.get("name", ""),
                    
                    "question": full_prompt,
                    
                    # LeetCode 数据集通常用于 Code Generation，
                    # 真正的 verification 需要 Sandbox 运行，
                    # 这里 true_answer 往往是空的，或者只有测试用例（如果有）
                    "true_answer": "", 
                    
                    "template": code_stub, # 保留原始模板，方便后续处理
                    "lang": self.lang,
                    "task": "LeetCode",
                    "file_name": rel_file_path
                }
                
                data_rows.append(data_row)
        
        self.data = pd.DataFrame(data_rows)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]
    
    def get_task_description(self):
        return (
        "You will solve a LeetCode algorithmic problem. "
        "Think step by step about the logic, time complexity, and edge cases. "
        "The last part of your response must be the Python solution code, "
        "strictly enclosed between the standard LeetCode markers as follows:\n\n"
        "# @lc code=start\n"
        "class Solution:\n"
        "    # Your code implementation here\n"
        "# @lc code=end\n\n"
        "Do not wrap this block in markdown code ticks (like ```python). "
        "Just output the markers and the code directly at the end."
    )