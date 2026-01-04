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

        # 1. Define subset list
        all_subsets = ["main", "socratic"]

        # 2. Determine subsets to load
        if name == "all":
            target_subsets = all_subsets
        elif name in all_subsets:
            target_subsets = [name]
        else:
            # Fault tolerance: if unknown name, default to main
            print(f"[Warning] Unknown subset '{name}'. Defaulting to 'main'.")
            target_subsets = ["main"]

        path = assemble_project_path(path)
        data_rows = []

        # 3. Iterate through subsets to load
        for subset_name in target_subsets:
            # Expected path: /data/gsm8k/test/main/metadata.jsonl
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
                    
                    # --- GSM8k specific answer parsing logic ---
                    # Original answer format example: "Janet sells ... <<9*2=18>>18.\n#### 18"
                    raw_answer = row.get("answer", "")
                    
                    reasoning_content = ""
                    final_answer = ""
                    
                    if "####" in raw_answer:
                        # Split by '####' symbol
                        parts = raw_answer.split("####")
                        # First part is reasoning process (CoT)
                        reasoning_content = parts[0].strip()
                        # Last part is final numeric answer (Gold Answer)
                        # strip() to remove potential newlines or spaces
                        final_answer = parts[-1].strip()
                    else:
                        # Exception handling: if no ####, use whole string as answer
                        final_answer = raw_answer.strip()

                    # --- Construct data row ---
                    # Ensure task_id is globally unique
                    raw_id = str(row.get("task_id", ""))
                    if raw_id:
                        unique_id = f"{subset_name}_{raw_id}"
                    else:
                        unique_id = f"{subset_name}_{len(data_rows)+1}"

                    data_row = {
                        "task_id": unique_id,
                        
                        "question": row.get("question", ""),
                        
                        # true_answer only stores final numeric value (e.g., "18")
                        # for easy exact match or numeric comparison during evaluation
                        "true_answer": final_answer,
                        
                        # save complete reasoning process for potential prompt learning
                        "reasoning": reasoning_content,
                        
                        "task": "GSM8k",
                        "subset": subset_name, # mark as main or socratic
                        "file_name": ""
                    }
                    
                    # Simple defense: remove commas from answer (e.g., "1,000" -> "1000")
                    # to improve numeric matching accuracy
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
