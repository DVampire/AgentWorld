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

        # 1. 定义已知子集列表
        # 根据你的描述，子集名称是全大写的
        all_subsets = ["GPQA_DIAMOND", "GPQA_EXTENDED", "GPQA_MAIN"]

        # 2. 确定要加载的目标子集
        if name == "all":
            target_subsets = all_subsets
        elif name in all_subsets:
            target_subsets = [name]
        else:
            # 容错：如果用户输入了小写，尝试匹配大写
            upper_name = name.upper()
            if upper_name in all_subsets:
                target_subsets = [upper_name]
            else:
                # 默认 fallback 到所有，或者你可以选择报错
                print(f"[Warning] Unknown subset '{name}'. Loading all GPQA subsets.")
                target_subsets = all_subsets

        path = assemble_project_path(path)
        data_rows = []
        
        # 固定随机种子，确保每次加载生成的选项顺序一致 (A/B/C/D 对应关系不变)
        rng = random.Random(42)

        # 3. 遍历子集加载数据
        for subset_name in target_subsets:
            # 假设目录结构: /path/split/GPQA_DIAMOND/metadata.jsonl
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
                    
                    # --- GPQA 特有的数据处理逻辑 ---

                    # 1. 获取基础信息
                    q_text = row.get("Question", "").strip()
                    correct_ans = row.get("Correct Answer", "").strip()
                    
                    # 2. 收集所有选项 (正确 + 3个错误)
                    # 注意：GPQA 包含 Incorrect Answer 1, 2, 3
                    choices = [correct_ans]
                    for i in range(1, 4):
                        wrong = row.get(f"Incorrect Answer {i}", "").strip()
                        if wrong:
                            choices.append(wrong)
                    
                    # 校验：如果选项不足2个，这题没法做选择题
                    if len(choices) < 2:
                        continue

                    # 3. 打乱选项 (Shuffle)
                    # 必须打乱，否则 A 永远是正确答案
                    rng.shuffle(choices)
                    
                    # 4. 寻找正确答案在打乱后列表中的位置，映射为 A-D
                    try:
                        correct_idx = choices.index(correct_ans)
                        correct_letter = chr(65 + correct_idx) # 0->A, 1->B ...
                    except ValueError:
                        continue

                    # 5. 构建带选项的 Prompt 文本
                    # 格式:
                    # [Question]
                    # A) [Option1]
                    # B) [Option2]
                    # ...
                    options_str_list = []
                    for idx, choice_text in enumerate(choices):
                        letter = chr(65 + idx)
                        options_str_list.append(f"{letter}) {choice_text}")
                    
                    full_question_prompt = f"{q_text}\n\n" + "\n".join(options_str_list) + "\nAnswer:"

                    # 6. 构造最终数据行
                    # 优先使用 Record ID，如果没有则生成
                    rec_id = row.get("Record ID", "")
                    if not rec_id:
                        rec_id = f"{subset_name}_{len(data_rows)+1}"
                        
                    data_row = {
                        "task_id": rec_id,
                        "question": full_question_prompt, # 已经是包含选项的完整 Prompt
                        "true_answer": correct_letter,    # 例如 "C"
                        "origin_answer": correct_ans,     # (可选) 保留原始答案文本用于 debug
                        "task": "GPQA",
                        "subset": subset_name,            # 记录来源于 DIAMOND/MAIN/EXTENDED
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