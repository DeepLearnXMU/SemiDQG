# coding=utf-8
from collections import Counter
import sys
import os
import jsonlines
from sacrebleu.metrics import BLEU
import numpy as np
from nltk.tokenize import word_tokenize


def distinct(seqs):
    """Calculate intra/inter distinct 1/2."""
    batch_size = len(seqs)
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams) + 1e-12) / (len(seq) + 1e-5))
        intra_dist2.append((len(bigrams) + 1e-12) / (max(0, len(seq) - 1) + 1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all) + 1e-12) / (sum(unigrams_all.values()) + 1e-5)
    inter_dist2 = (len(bigrams_all) + 1e-12) / (sum(bigrams_all.values()) + 1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return intra_dist1, intra_dist2, inter_dist1, inter_dist2


# macro F1 score
def uni_F1_score(preds, labels):
    f1_scores = []
    for pred, label in zip(preds, labels):
        pred = lang_tokenize(pred)

        def scoring(x, y):
            x, y = set(x), set(y)
            x_len = len(x)
            y_len = len(y)
            share_len = len(x & y)
            try:
                p, r = share_len / x_len, share_len / y_len
            except:
                p, r = 0, 0
            if p == 0 or r == 0:
                return 0
            else:
                return 2 * p * r / (p + r)

        if isinstance(label, list):
            f1_scores.append(max([scoring(pred, lang_tokenize(ref)) for ref in label]))
        else:
            f1_scores.append(scoring(pred, lang_tokenize(label)))
    return np.mean(f1_scores)


if __name__ == "__main__":
    lang = sys.argv[1]
    assert lang in ["en", "zh"]
    data_folder = "data_woi" if lang == "en" else "data_dusinc"
    predict_folder = sys.argv[2]
    mode = "test"
    if len(sys.argv) > 3:
        mode = sys.argv[3]
    assert mode in ["valid", "test", "train"]
    if mode == "test":
        data_path = f"../saved_data/{data_folder}/test.json"
    elif mode == "valid":
        data_path = f"../saved_data/{data_folder}/valid.json"
    else:
        data_path = f"../saved_data/{data_folder}/train.json"
    if mode == "test":
        predict_path = os.path.join(predict_folder, "generated_predictions.txt")
    elif mode == "valid":
        predict_path = os.path.join(predict_folder, "generated_predictions_valid.txt")
    else:
        predict_path = os.path.join(predict_folder, "generated_predictions_train.txt")
    if lang == "en":
        bleu1 = BLEU(max_ngram_order=1)
        bleu2 = BLEU(max_ngram_order=2)
        lang_tokenize = word_tokenize
    elif lang == "zh":
        bleu1 = BLEU(tokenize="zh", max_ngram_order=1)
        bleu2 = BLEU(tokenize="zh", max_ngram_order=2)
        lang_tokenize = lambda x: x

    preds = []
    labels = []
    max_ref_num = 0
    with jsonlines.open(data_path, "r") as reader:
        for line in reader:
            if isinstance(line["query"], str):
                labels.append([line["query"].lower()])
            else:
                labels.append([x.lower() for x in line["query"]])
            max_ref_num = max(max_ref_num, len(line["query"]))

    with open(predict_path, "r") as f:
        for i, line in enumerate(f.readlines()):
            preds.append(line.strip().lower())

    bleu_labels = []
    for i in range(max_ref_num):
        bleu_labels.append([])
        for j in range(len(labels)):
            if len(labels[j]) > i:
                bleu_labels[i].append(labels[j][i])
            else:
                bleu_labels[i].append(labels[j][-1])

    b1_score = bleu1.corpus_score(preds, bleu_labels).score
    b2_score = bleu2.corpus_score(preds, bleu_labels).score
    f1_score = uni_F1_score(preds, labels) * 100
    print(
        f"BLEU-1/2 {round(b1_score, 2)} / {round(b2_score, 2)}\nUni.F1 {round(f1_score, 2)}"
    )
    print(f"summation {round(b1_score + b2_score + f1_score, 2)}")
