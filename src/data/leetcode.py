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

        # 1. Path normalization
        #path = assemble_project_path(path)
        
        # 2. Load metadata file
        # Expected path structure: /data/leetcode/test/metadata.jsonl
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
                
                # --- LeetCode specific loading logic ---

                # 1. Read problem description (Markdown content)
                # row['file'] example: "./question/1.two_sum.md"
                # Parse to absolute path, usually relative to metadata.jsonl directory
                rel_file_path = row.get("file", "")
                question_content = ""
                
                if rel_file_path:
                    # Remove potential "./" prefix and join with full path
                    # Assume question folder and metadata.jsonl are in the same split directory
                    clean_rel_path = rel_file_path.replace("./", "") 
                    abs_file_path = os.path.join(path, split, clean_rel_path)
                    
                    if os.path.exists(abs_file_path):
                        with open(abs_file_path, "r", encoding="utf-8") as qf:
                            question_content = qf.read()
                    else:
                        print(f"[Warning] Question file not found: {abs_file_path}")
                        # If description file is missing, skip this problem
                        continue
                
                # 2. Get code template for specified language
                templates = row.get("code_template", {})
                # Get user specified language, fallback to python3 or first available
                code_stub = templates.get(self.lang, "")
                if not code_stub and self.lang == "python3":
                    # Try python (legacy field)
                    code_stub = templates.get("python", "")
                
                # 3. Build Prompt
                # Typical code generation Prompt: description + code stub
                full_prompt = f"Problem Description:\n{question_content}\n\n" \
                              f"Please write a solution in {self.lang}:\n```\n{code_stub}\n```"

                # 4. Construct data row
                data_row = {
                    "task_id": str(row.get("id", "")), # to string
                    
                    "name": row.get("name", ""),
                    
                    "question": full_prompt,
                    
                    # LeetCode datasets are typically for Code Generation,
                    # real verification requires Sandbox execution,
                    # true_answer is usually empty or contains test cases (if any)
                    "true_answer": "", 
                    
                    "template": code_stub, # keep original template for later processing
                    "lang": self.lang,
                    "task": "LeetCode",
                    "file_name": rel_file_path
                }
                
                data_rows.append(data_row)
        
        # 4. Convert to DataFrame
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