import os
import json
import pandas as pd

from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(force=True)
class AIME25Dataset:
    def __init__(self, path, name, split, **kwargs):
        """
        Initialize AIME 2025 Dataset.
        
        Args:
            path: Base path to the dataset directory
            name: Dataset name (Not used for filtering, kept for compatibility)
            split: Dataset split ("test", "validation", etc.)
        """
        self.path = path
        self.name = name
        self.split = split

        # 1. Path processing
        base_path = assemble_project_path(path)
        split_path = os.path.join(base_path, split)
        
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split directory not found: {split_path}")
        
        # 2. Find all metadata.jsonl files in subdirectories
        # Structure: datasets/AIME25/test/AIME2025-I/metadata.jsonl, AIME2025-II/metadata.jsonl, etc.
        data_rows = []
        
        # Traverse all subdirectories in the split directory
        for item in os.listdir(split_path):
            item_path = os.path.join(split_path, item)
            if os.path.isdir(item_path):
                metadata_file = os.path.join(item_path, "metadata.jsonl")
                if os.path.exists(metadata_file):
                    # Read data from this metadata.jsonl
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            q_text = row.get("problem") or row.get("question")
                            a_text = row.get("answer")
                            
                            if q_text:
                                unique_id = str(row.get("task_id", f"aime25_{len(data_rows)}"))

                                data_row = {
                                    "task_id": unique_id,
                                    "question": q_text,
                                    "true_answer": str(a_text) if a_text is not None else "",
                                    "task": "AIME 2025"
                                }
                                
                                # Defensive answer handling: handle numeric answer in json
                                if isinstance(data_row["true_answer"], (int, float)):
                                    data_row["true_answer"] = str(int(data_row["true_answer"]))
                                
                                data_rows.append(data_row)
        
        # 3. Convert to DataFrame
        self.data = pd.DataFrame(data_rows)
        
        if len(self.data) == 0:
            print(f"Warning: No data loaded from {split_path}")
        else:
            print(f"Successfully loaded {len(self.data)} tasks from {split_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]