import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import jsonlines
from tqdm import tqdm

# from nltk import word_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# from nltk.corpus import stopwords
# from string import punctuation

# func_words = set([w for w in stopwords.words('english')]) | set(punctuation)

model_path = f"../../ckpt/t5-base-wow-p"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    truncation_side="left",
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()
bsz = 16
num_seqs = 15
posterior = False
src_file = f"../../saved_data/data_wow/train.json"
tgt_file = f"../../data/data_wow/train_with_wow_p.json"

with jsonlines.open(src_file, "r") as reader:
    data = [line for line in reader]

writer = jsonlines.open(tgt_file, "w")

print(
    model_path,
    "bsz:",
    bsz,
    "num_seqs:",
    num_seqs,
    "posterior:",
    posterior,
    "src_file:",
    src_file,
    "tgt_file:",
    tgt_file,
)

pbar = tqdm(total=len(data))
for idx in range(0, len(data), bsz):
    inputs = []
    for example in data[idx : idx + bsz]:
        input = (
            f"{example['dialogue']}\n{example['response']}"
            if posterior
            else example["dialogue"]
        )
        inputs.append(input.lower())
    model_inputs = tokenizer(
        inputs, return_tensors="pt", max_length=256, truncation=True, padding=True
    )
    model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

    predictions = model.generate(
        **model_inputs, num_beams=num_seqs, num_return_sequences=num_seqs, max_length=50
    )
    predictions = tokenizer.batch_decode(
        predictions, clean_up_tokenization_spaces=True, skip_special_tokens=True
    )
    assert len(inputs) * num_seqs == len(predictions)

    for sub_idx, example in enumerate(data[idx : idx + bsz]):
        example["queries"] = predictions[
            sub_idx * num_seqs : sub_idx * num_seqs + num_seqs
        ]
    pbar.update(len(inputs))

writer.write_all(data)
