import jsonlines
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tqdm import tqdm

lang = "zh"
threshold = 0.3
src_file = "../../data/data_kdconv/train_with_kdconv_q.json"
ref_file = "../../data/data_kdconv/train_with_kdconv_p.json"
tgt_file = "../../data/data_kdconv/train_with_kdconv_qc03.json"
topk = 1
use_rep_penalty = True

print("lang", lang)
print("threshold", threshold)
print("src_file", src_file)
print("ref_file", ref_file)
print("tgt_file", tgt_file)
print("topk", topk)
print("use_rep_penalty", use_rep_penalty)

with jsonlines.open(src_file, "r") as f:
    s_queries = [line for line in f]

with jsonlines.open(ref_file, "r") as f:
    t_queries = [line for line in f]


ps = PorterStemmer()
beta = 1  # 1 for F1


def remove_freq_word(words):
    return set(words)


def tokenize(str, lang="en"):
    return (
        list(str.lower())
        if lang == "zh"
        else [ps.stem(w) for w in word_tokenize(str.lower())]
    )


def eval(pred, ref):
    pred = tokenize(pred, lang=lang)
    ref = tokenize(ref, lang=lang)
    pred_len = len(pred)
    ref_len = len(ref)
    pred_set = remove_freq_word(pred)
    ref_set = remove_freq_word(ref)
    share_len = len(pred_set & ref_set)
    try:
        p, r = share_len / pred_len, share_len / ref_len
        rep_penalty = (
            min((len(pred_set) / pred_len) / (len(ref_set) / ref_len), 1.0) ** 0.5
            if use_rep_penalty
            else 1.0
        )
        return (1 + beta**2) * p * r / ((beta**2) * p + r) * rep_penalty
    except:
        return 0.0


def cons_filter(t_queries, s_queries, threshold=0.5):
    assert len(t_queries) == len(s_queries)
    data = []
    for i, line in tqdm(enumerate(t_queries)):
        for s_query in s_queries[i]["queries"][:topk]:
            if eval(t_queries[i]["queries"][0], s_query) >= threshold:
                data.append(line)
                break
    print("all:", len(t_queries))
    print("filtered:", len(data))
    return data


sc_queries = cons_filter(s_queries, t_queries, threshold=threshold)
with jsonlines.open(tgt_file, "w") as f:
    f.write_all(sc_queries)
