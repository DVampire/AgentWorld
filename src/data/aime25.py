import os
import json
import pandas as pd

from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(force=True)
class AIME25Dataset:
    # Core change: use **kwargs to absorb all uncertain parameters (like subset, name, etc.)
    def __init__(self, path, split, **kwargs):
        """
        Args:
            path: Dataset root directory (e.g., ./datasets/AIME25)
            split: Data split (e.g., test)
        """
        self.path = path
        self.split = split
        
        # 1. Convert path and print debug info
        base_path = assemble_project_path(path)
        
        # 2. Auto-locate metadata.jsonl
        # Logic: search for metadata.jsonl in {path}/{split}/
        metadata_file = os.path.join(base_path, split, "metadata.jsonl")
        
        # Debug: check if this path exists on your computer
        print(f"DEBUG: Dataset searching at -> {os.path.abspath(metadata_file)}")
        
        data_rows = []

        if not os.path.exists(metadata_file):
            print(f"Error: File not found at {metadata_file}")
            self.data = pd.DataFrame()
            return

        # 3. Read data
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
        print(f"Successfully loaded {len(self.data)} tasks.")
    
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