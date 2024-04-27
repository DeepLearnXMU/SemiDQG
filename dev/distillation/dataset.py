# coding=utf-8
import random

import jsonlines
from torch.utils.data import Dataset
import numpy as np


class WoIDataset(Dataset):
    def __init__(self, file_path, topk=1, posterior=False):
        self.topk = topk
        self.posterior = posterior
        print(f"Prepare data from {file_path}")
        with jsonlines.open(file_path, "r") as reader:
            self.data = [self.preprocess(line) for line in reader]

    def preprocess(self, example):
        return {
            "dialogue": example["dialogue"].lower(),
            "response": example["response"].lower(),
            "query": example["query"].lower(),
            "queries": [query.lower() for query in example["queries"]]
            if "queries" in example
            else [example["query"].lower()],
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        source = (
            example["dialogue"]
            if not self.posterior
            else f"{example['dialogue']}\n{example['response']}"
        )
        post_source = f"{example['dialogue']}\n{example['response']}"
        reference = example["query"]
        if "queries" in example and example["queries"]:
            target = random.choice(example["queries"][: self.topk])
        else:
            target = None
        return source, post_source, target, reference
