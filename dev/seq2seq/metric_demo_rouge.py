# from datasets import load_metric
import evaluate
import nltk
import os
import jieba
import jsonlines
import sacrebleu as scb

# from sacrebleu.metrics import BLEU
import numpy as np
from nltk.tokenize import word_tokenize


# metric = load_metric("rouge")
metric = evaluate.load("rouge")
lang = "zh"

data_path = "../../saved_data/data_kdconv/train.json"
predict_path = os.path.join("../../qp_top15_best.txt")

max_ref_num = 0

labels, preds = [], []

with jsonlines.open(data_path, "r") as reader:
    for line in reader:
        if isinstance(line["query"], str):
            labels.append([line["query"].lower()])
        else:
            labels.append([x.lower() for x in line["query"]])
        max_ref_num = max(max_ref_num, len(line["query"]))
max_ref_num = 1
with open(predict_path, "r") as f:
    for i, line in enumerate(f.readlines()):
        preds.append(line.strip().lower())
print("max_ref_num:", max_ref_num)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    print(preds[:5])
    print(labels[:5])
    return preds, labels


decoded_preds, decoded_labels = postprocess_text(preds, [l[0] for l in labels])


class Tokenizer:
    """Helper class to wrap a callable into a class with a `tokenize` method as used by rouge-score."""

    def __init__(self, tokenizer_func):
        self.tokenizer_func = tokenizer_func

    def tokenize(self, text):
        return list(text)
        # return self.tokenizer_func(text)


result = metric.compute(
    predictions=decoded_preds,
    references=decoded_labels,
    tokenizer=(lambda x: list(x)) if lang == "zh" else None,
)

print(result)
