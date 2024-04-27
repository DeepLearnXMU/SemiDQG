import json
import jsonlines
import os
import threading
import numpy as np
import wikipedia
from tqdm import tqdm

pred_path = "../saved_data/data_wow/valid_with_p_all_queries.json"

use_topk_pred = 5
pred_set = set()
preds, refs = [], []
with jsonlines.open(pred_path, "r") as reader:
    for line in reader:
        preds.append(line["queries"][:use_topk_pred])
        pred_set.update(set(line["queries"][:use_topk_pred]))
        refs.append(line["passage"])

output_path = "../saved_data/data_wow/query_search_for_valid_with_p_all_queries.json"

if os.path.exists(output_path):
    with open(output_path, "r") as f:
        pdkp2tt = json.load(f)
else:
    pdkp2tt = {}

kps = pred_set - set(pdkp2tt.keys())
print(f"{len(kps)} query need searching..")

pbar = tqdm(len(kps))
data_iter = iter(kps)


def thread_fn(name):
    global cnt
    global total
    global data_iter
    while True:
        try:
            kp = next(data_iter)
        except StopIteration:
            break
        try:
            pdkp2tt[kp] = wikipedia.search(kp, results=5)
        except:
            pass
        pbar.update()


thread_num = 10
threads = []
for i in range(thread_num):
    new_thread = threading.Thread(target=thread_fn, args=("Thread-{}".format(i),))
    new_thread.start()
    threads.append(new_thread)
for thread in threads:
    thread.join()

pbar.close()

with open(output_path, "w") as f:
    json.dump(pdkp2tt, f)


def recall_k(preds, refs):
    rank = []
    for pred, ref in zip(preds, refs):
        cur_rank = 100
        for i, _pred in enumerate(pred):
            if _pred in pdkp2tt:
                if ref in pdkp2tt[_pred]:
                    cur_rank = i
                    break
        rank.append(cur_rank)
    rank = np.array(rank)
    return np.mean(rank < 1), np.mean(rank < 3), np.mean(rank < 5)


print(recall_k(preds, refs))
