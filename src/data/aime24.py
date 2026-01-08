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

        # 1. Path processing
        path = assemble_project_path(path)
        
        # 2. Locate metadata.jsonl file
        metadata_file = os.path.join(path, split, "metadata.jsonl")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # 3. Read and clean data
        data_rows = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue # Skip corrupted lines

                # --- For data format {"task_id": 60, "problem": "...", "answer": "204"} ---
                
                # Extract fields
                # Note: original task_id is integer 60, convert to string for consistency
                t_id = str(row.get("task_id", ""))
                q_text = row.get("problem", "")
                a_text = row.get("answer", "")
                
                # Simple validation: valid if problem text exists
                if q_text:
                    data_row = {
                        "task_id": t_id,          # maps to task_id in json
                        "question": q_text,       # map problem to question
                        "true_answer": a_text,    # map answer to true_answer
                        "task": "AIME 2024",      # fixed tag for source identification
                        "file_name": ""           # AIME problems are usually text-only
                    }
                    
                    # Defensive answer handling: handle numeric answer in json
                    if isinstance(data_row["true_answer"], (int, float)):
                        data_row["true_answer"] = str(int(data_row["true_answer"]))
                        
                    data_rows.append(data_row)
        
        # 4. Convert to DataFrame
        self.data = pd.DataFrame(data_rows)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]