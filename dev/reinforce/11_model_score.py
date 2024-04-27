import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import jsonlines
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

model_path = f""
src_file = f""
tgt_file = f""
print("model_path:", model_path, "src_file:", src_file, "tgt_file:", tgt_file)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    truncation_side="left",
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).cuda()


def cal_prob(example):
    input_strs = [f"""{example["dialogue"]}\n{example["response"]}""".lower()] * len(
        example["queries"]
    )
    target_strs = example["queries"]
    model_inputs = tokenizer(
        input_strs, return_tensors="pt", max_length=256, truncation=True, padding=True
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_strs,
            max_length=64,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
    labels = labels["input_ids"]
    # prepare decoder_input_ids
    decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
    model_inputs["decoder_input_ids"] = decoder_input_ids
    model_inputs = {k: v.cuda() for k, v in model_inputs.items()}

    model_outputs = model(**model_inputs)
    lm_logits = model_outputs.logits
    lm_probs = F.softmax(lm_logits, dim=-1)
    vocab_size = lm_probs.shape[-1]
    labels[labels < 0] = tokenizer.pad_token_id  # -100 --> pad
    label_mask = F.one_hot(labels, num_classes=vocab_size)
    flat_probs = lm_probs[label_mask.bool()]
    probs = flat_probs.reshape_as(labels)
    probs[labels == tokenizer.pad_token_id] = 1.0  # notice pad
    len_norm = (
        (5 + torch.sum(labels != tokenizer.pad_token_id, dim=-1)) / (5 + 1)
    ) ** 0.5 + 0.5
    # q_prob = torch.softmax(torch.sum(torch.log(probs), dim=-1), dim=-1)
    q_probs = torch.softmax(
        torch.sum(torch.log(probs), dim=-1) / len_norm.to(probs), dim=-1
    )
    return {k: round(v, 6) for k, v in zip(target_strs, q_probs.tolist())}


data = []
pbar = tqdm()
with jsonlines.open(src_file, "r") as reader:
    for line in reader:
        scored_queries = cal_prob(line)
        data.append(
            {
                "dialogue": line["dialogue"],
                "response": line["response"],
                "queries": scored_queries,
                "target": line["query"] if "query" in line else None,
            }
        )
        pbar.update()

with jsonlines.open(tgt_file, "w") as writer:
    writer.write_all(data)
