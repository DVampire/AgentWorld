import platformdirs

from .base import Dataset


def _split_items(items, split):
    if split not in ["train", "val", "test", "all"]:
        raise ValueError(f"Unsupported split '{split}'. Expected one of ['train', 'val', 'test', 'all'].")
    if len(items) < 10:
        raise ValueError("AIME dataset expects at least 10 records to build splits.")
    if split == "train":
        return items[:20]
    if split == "val":
        return items[20:25]
    if split == "all":
        return items
    return items[25:]


class AIME24(Dataset):
    def __init__(self, root: str = None, split: str = "train", *args, **kwargs):
        from datasets import load_dataset

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        raw_data = load_dataset("HuggingFaceH4/aime_2024", split="train", cache_dir=root, *args, **kwargs)
        items = []
        for row in raw_data:
            items.append(
                dict(
                    question=row["problem"].strip(),
                    answer=str(row["answer"]).strip(),
                    solution=row.get("solution", "").strip(),
                )
            )
        self.data = _split_items(items, split)
        self.split = split
        self.root = root

    def __getitem__(self, index):
        item = self.data[index]
        question = f"Question: {item['question']}"
        return question, item["answer"]

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return (
            "You will answer a challenging mathematics contest problem. Think step by step. "
            "State intermediate reasoning clearly. The last line of your response should be "
            "of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
        )


class AIME25(Dataset):
    def __init__(self, root: str = None, split: str = "train", *args, **kwargs):
        from datasets import load_dataset

        if root is None:
            root = platformdirs.user_cache_dir("textgrad")
        subset_names = ["AIME2025-I", "AIME2025-II"]
        items = []
        for subset in subset_names:
            try:
                raw_subset = load_dataset(
                    "opencompass/AIME2025",
                    subset,
                    split="train",
                    cache_dir=root,
                    *args,
                    **kwargs,
                )
            except ValueError:
                raw_subset = load_dataset(
                    "opencompass/AIME2025",
                    subset,
                    split="test",
                    cache_dir=root,
                    *args,
                    **kwargs,
                )
            for row in raw_subset:
                items.append(
                    dict(
                        question=row["question"].strip(),
                        answer=str(row["answer"]).strip(),
                        subset=subset,
                    )
                )
        self.data = _split_items(items, split)
        self.split = split
        self.root = root

    def __getitem__(self, index):
        item = self.data[index]
        question = f"Question: {item['question']}"
        return question, item["answer"]

    def __len__(self):
        return len(self.data)

    def get_task_description(self):
        return (
            "You will answer a challenging mathematics contest problem. Think step by step. "
            "State intermediate reasoning clearly. The last line of your response should be "
            "of the following format: 'Answer: $VALUE' where VALUE is a numerical value."
        )
