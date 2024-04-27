import os
import torch
import numpy as np

from transformers import PreTrainedModel
from transformers.modeling_utils import unwrap_model
from nltk.tokenize import word_tokenize


def uni_F1_score(preds, labels):
    f1_score = []
    for pred, label in zip(preds, labels):
        pred_len = len(pred)
        label_len = len(label)
        common_len = len(set(pred) & set(label))
        try:
            p, r = common_len / pred_len, common_len / label_len
        except:
            p, r = 0, 0
        if p == 0 or r == 0:
            _f1_score = 0
        else:
            _f1_score = 2 * p * r / (p + r)
        f1_score.append(_f1_score)
    return np.mean(f1_score)


def batch2gpu(batch):
    return {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


WEIGHTS_NAME = "pytorch_model.bin"
TRAINING_ARGS_NAME = "training_args.bin"


def save_model(args=None, model=None, tokenizer=None, state_dict=None, step=None):
    if step:
        output_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model checkpoint to {output_dir}")
    # Save a trained model and configuration using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    if not isinstance(model, PreTrainedModel):
        if isinstance(unwrap_model(model), PreTrainedModel):
            if state_dict is None:
                state_dict = model.state_dict()
            unwrap_model(model).save_pretrained(output_dir, state_dict=state_dict)
        else:
            print(
                "Trainer.model is not a `PreTrainedModel`, only saving its state dict."
            )
            if state_dict is None:
                state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
    else:
        model.save_pretrained(output_dir, state_dict=state_dict)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, TRAINING_ARGS_NAME))
