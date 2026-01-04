import os
import json
import random
import pandas as pd

from src.registry import DATASET
from src.utils import assemble_project_path


@DATASET.register_module(force=True)
class GPQADataset:
    def __init__(self, path, name, split):
        """
        Initialize GPQA Dataset with Diamond/Extended/Main support.
        
        Args:
            path: Base path to the dataset directory
            name: "all" or specific subset ("GPQA_DIAMOND", "GPQA_EXTENDED", "GPQA_MAIN")
            split: Dataset split ("test", "validation", "train")
        """
        self.path = path
        self.name = name
        self.split = split

        # 1. Define known subset list
        all_subsets = ["GPQA_DIAMOND", "GPQA_EXTENDED", "GPQA_MAIN"]

        # 2. Determine target subsets to load
        if name == "all":
            target_subsets = all_subsets
        elif name in all_subsets:
            target_subsets = [name]
        else:
            # Fault tolerance: try matching upper case if input is lower case
            upper_name = name.upper()
            if upper_name in all_subsets:
                target_subsets = [upper_name]
            else:
                # Default fallback to all
                print(f"[Warning] Unknown subset '{name}'. Loading all GPQA subsets.")
                target_subsets = all_subsets

        path = assemble_project_path(path)
        data_rows = []
        
        # Fixed random seed to ensure consistent option order (A/B/C/D mapping)
        rng = random.Random(42)

        # 3. Iterate through subsets to load data
        for subset_name in target_subsets:
            # Expected directory structure: /path/split/GPQA_DIAMOND/metadata.jsonl
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
                    
                    # --- GPQA specific data processing logic ---

                    # 1. Get basic info
                    q_text = row.get("Question", "").strip()
                    correct_ans = row.get("Correct Answer", "").strip()
                    
                    # 2. Collect all choices (Correct + 3 Incorrect)
                    # Note: GPQA contains Incorrect Answer 1, 2, 3
                    choices = [correct_ans]
                    for i in range(1, 4):
                        wrong = row.get(f"Incorrect Answer {i}", "").strip()
                        if wrong:
                            choices.append(wrong)
                    
                    # Validation: skip if fewer than 2 choices
                    if len(choices) < 2:
                        continue

                    # 3. Shuffle choices
                    # Must shuffle, otherwise A will always be correct
                    rng.shuffle(choices)
                    
                    # 4. Find position of correct answer in shuffled list, map to A-D
                    try:
                        correct_idx = choices.index(correct_ans)
                        correct_letter = chr(65 + correct_idx) # 0->A, 1->B ...
                    except ValueError:
                        continue

                    # 5. Construct Prompt text with options
                    # Format:
                    # [Question]
                    # A) [Option1]
                    # B) [Option2]
                    # ...
                    options_str_list = []
                    for idx, choice_text in enumerate(choices):
                        letter = chr(65 + idx)
                        options_str_list.append(f"{letter}) {choice_text}")
                    
                    full_question_prompt = f"{q_text}\n\n" + "\n".join(options_str_list) + "\nAnswer:"

                    # 6. Construct final data row
                    # Use Record ID if available, otherwise generate one
                    rec_id = row.get("Record ID", "")
                    if not rec_id:
                        rec_id = f"{subset_name}_{len(data_rows)+1}"
                        
                    data_row = {
                        "task_id": rec_id,
                        "question": full_question_prompt, # complete prompt containing choices
                        "true_answer": correct_letter,    # e.g., "C"
                        "origin_answer": correct_ans,     # (optional) keep original answer for debugging
                        "task": "GPQA",
                        "subset": subset_name,            # record source DIAMOND/MAIN/EXTENDED
                        "subdomain": row.get("Subdomain", ""),
                        "file_name": ""
                    }
                    
                    data_rows.append(data_row)
        
        self.data = pd.DataFrame(data_rows)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.iloc[index]
    def get_task_description(self):
        return """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()