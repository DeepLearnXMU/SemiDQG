# coding=utf-8
import random

import jsonlines
from torch.utils.data import Dataset
import numpy as np


class QueryGenDataset(Dataset):
    def __init__(self, file_path, posterior=True):
        self.posterior = posterior
        with jsonlines.open(file_path, "r") as reader:
            self.data = [self.preprocess(line) for line in reader]

    def preprocess(self, example):
        return {
            "dialogue": example["dialogue"].lower(),
            "response": example["response"].lower(),
            "query": example["query"].lower(),
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        return (
            f"{example['dialogue']}\n{example['response']}"
            if self.posterior
            else example["dialogue"],
            example["query"],
        )


class RAQGDataset(Dataset):
    def __init__(
        self,
        file_path,
        topk=-1,
        threshold=0.3,
        posterior=True,
        beam_top_filter=False,
        score_normalization=True,
    ):
        self.topk, self.threshold, self.posterior, self.score_normaliztion = (
            topk,
            threshold,
            posterior,
            score_normalization,
        )
        print(f"Prepare data from {file_path}")
        with jsonlines.open(file_path, "r") as reader:
            self.data = [self.preprocess(line) for line in reader]
        print(f"Before filter : {len(self.data)}")
        self.data = [
            x
            for x in [
                self.filter(x, beam_top_filter=beam_top_filter) for x in self.data
            ]
            if x
        ]
        print(f"After filter : {len(self.data)}")

    def preprocess(self, example):
        return {
            "dialogue": example["dialogue"].lower(),
            "response": example["response"].lower(),
            "query": example["target"].lower()
            if "target" in example
            else example["query"],
            "queries": {k.lower(): v for k, v in example["queries"].items()},
        }

    def filter(self, example, beam_top_filter=False):
        query_dict = example["queries"]
        if beam_top_filter:
            queries = list(query_dict.keys())[: self.topk]
        else:
            queries = sorted(query_dict, key=lambda x: query_dict[x], reverse=True)[
                : self.topk
            ]
        scores = [query_dict[query] for query in queries]
        if (
            scores[0] >= self.threshold and max(scores) - min(scores) > 0.01
        ) or self.topk == 1:
            example["queries"], example["scores"] = queries, scores
            return example
        return None

    def normalize(self, scores):
        scores = np.array(scores)
        mean_score = np.mean(scores)
        scores = scores - mean_score
        return scores.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        targets = example["queries"]
        inputs = [
            f"{example['dialogue']}\n{example['response']}"
            if self.posterior
            else example["dialogue"]
        ] * len(targets)
        scores = (
            self.normalize(example["scores"])
            if self.score_normaliztion
            else example["scores"]
        )
        return inputs, targets, scores
