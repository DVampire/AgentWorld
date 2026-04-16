import os
import json
import pandas as pd

from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(force=True)
class ESGDataset:
    def __init__(self, path, name, split):
        """
        Initialize ESG Dataset.
        
        Args:
            path: Base path to the ESG dataset directory
            name: Dataset name, format: "all" or "level1", "level2", "level3"
            split: Dataset split ("test", "validation", "train", etc.)
        """
        self.path = path
        self.name = name
        self.split = split

        path = assemble_project_path(path)
        
        # Parse name to get levels
        # name format: "all" or "level1", "level2", "level3"
        if name == "all":
            levels = [1, 2, 3]
        elif name.startswith("level"):
            levels = [int(name.replace("level", ""))]
        else:
            levels = [1, 2, 3]  # default to all levels
        
        # Load metadata.jsonl file
        metadata_file = os.path.join(path, split, "metadata.jsonl")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Read and filter data
        data_rows = []
        with open(metadata_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                # Filter by level
                if int(row.get("Level", 0)) in levels:
                    # Rename columns
                    data_row = {
                        "task_id": row.get("task_id", ""),
                        "question": row.get("Question", ""),
                        "true_answer": row.get("Final answer", "?"),
                        "task": row.get("Level", ""),
                        "file_name": row.get("file_name", ""),
                    }
                    # Process file path
                    if data_row["file_name"]:
                        file_path = os.path.join(path, split, data_row["file_name"])
                        if os.path.exists(file_path):
                            data_row["file_name"] = file_path
                        else:
                            # File doesn't exist, set to empty
                            data_row["file_name"] = ""
                    data_rows.append(data_row)
        
        self.data = pd.DataFrame(data_rows)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]