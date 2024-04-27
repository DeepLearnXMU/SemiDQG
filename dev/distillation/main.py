import importlib
import sys
import math
import torch

torch.autograd.set_detect_anomaly(True)

from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from torch.utils.data import DataLoader
from dataset import WoIDataset
from utils import *
from model import T5ForConditionalGenerationKD


def main(args):
    print(args.__dict__)
    # Set random seed
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, truncation_side="left"
    )
    model = T5ForConditionalGenerationKD.from_pretrained(args.model_name_or_path)
    model = model.cuda()
    setattr(model.config, "alpha", args.alpha)
    setattr(model.config, "beta", args.beta)
    setattr(model.config, "gamma", args.gamma)

    def collate_fn(batch):
        sources, post_sources, targets, references = zip(*batch)
        sources, post_sources, targets, references = (
            list(sources),
            list(post_sources),
            list(targets),
            list(references),
        )

        model_inputs = tokenizer(
            sources,
            max_length=args.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        post_model_inputs = tokenizer(
            post_sources,
            max_length=args.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if targets[0]:
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=args.max_target_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                labels["input_ids"][
                    labels["input_ids"] == tokenizer.pad_token_id
                ] = -100
            model_inputs["labels"] = labels["input_ids"]
            post_model_inputs["labels"] = labels["input_ids"]

            decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
                labels=model_inputs["labels"]
            )
            model_inputs["decoder_input_ids"] = decoder_input_ids
            post_model_inputs["decoder_input_ids"] = decoder_input_ids

        return model_inputs, post_model_inputs, references

    def eval_collate_fn(batch):
        sources, post_sources, targets, references = zip(*batch)
        sources, post_sources, targets, references = (
            list(sources),
            list(post_sources),
            list(targets),
            list(references),
        )

        model_inputs = tokenizer(
            sources,
            max_length=args.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return model_inputs, references

    train_dataset = WoIDataset(args.train_file, args.topk, args.posterior)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    valid_dataset = WoIDataset(args.valid_file, args.topk, args.posterior)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eval_collate_fn,
    )

    test_dataset = WoIDataset(args.test_file, args.topk, args.posterior)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=eval_collate_fn,
    )

    num_training_steps = (
        len(train_dataloader) * args.epoches // args.gradient_accumulation_steps
        if args.max_steps is None
        else args.max_steps
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def evaluate(model):
        predictions, references = [], []
        model.eval()
        pbar = tqdm(total=len(valid_dataloader))
        for inputs, refs in valid_dataloader:
            inputs = batch2gpu(inputs)
            with torch.no_grad():
                pred_tokens = model.generate(
                    **inputs, num_beams=4, max_length=args.max_target_length
                )
            preds = tokenizer.batch_decode(
                pred_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True
            )
            predictions += preds
            references += refs
            pbar.update()
        if args.lang == "en":

            def preprocess(strs):
                return [word_tokenize(str) for str in strs]

            predictions, references = preprocess(predictions), preprocess(references)
        return uni_F1_score(predictions, references)

    if args.do_train:
        backward_steps, global_steps = 0, 0
        train_loss = 0.0
        model.train()
        if args.max_steps:
            if args.save_steps is None:
                print("`save_steps` not found")
                exit(-1)
            pbar = tqdm(total=args.max_steps)
            epoches = (
                args.max_steps
                * args.gradient_accumulation_steps
                / len(train_dataloader)
            )
            print("Total epochs:", epoches)
        else:
            pbar = tqdm(
                total=len(train_dataloader)
                * args.epoches
                // args.gradient_accumulation_steps
            )
            epoches = args.epoches
        pbar = tqdm(total=num_training_steps)
        for epoch in range(math.ceil(epoches)):
            pbar.set_description(f"Epoch {epoch + 1} ")
            model.train()
            for inputs, post_inputs, refs in train_dataloader:
                inputs = batch2gpu(inputs)
                post_inputs = batch2gpu(post_inputs)
                if args.mode == "vanilla-kd":
                    loss = model(**inputs).loss
                else:
                    raise Exception("Unknown Training Mode...")
                if loss is not None:
                    pass
                else:
                    continue
                loss /= args.gradient_accumulation_steps
                train_loss += loss.item()
                loss.backward()
                backward_steps += 1
                if backward_steps % args.gradient_accumulation_steps == 0:
                    pbar.update(1)
                    pbar.set_description(f"Epoch {epoch+1} Loss {round(train_loss,3)} ")
                    global_steps += 1
                    train_loss = 0.0
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    if args.max_steps and global_steps % args.save_steps == 0:
                        eval_score = evaluate(model)
                        print(
                            f"Step {global_steps}, Evaluation score {round(eval_score, 4)}"
                        )
                        save_model(args, model, tokenizer, step=global_steps)
                        print(f"saving model at checkpoint step {global_steps}")
                        model.train()
                    if global_steps == args.max_steps:
                        break
            if args.max_steps is None:
                eval_score = evaluate(model)
                print(f"Step {global_steps}, Evaluation score {round(eval_score, 4)}")
                save_model(args, model, tokenizer, step=global_steps)
                print(f"saving model at checkpoint step {global_steps}")
                model.train()
        pbar.close()

    if args.do_predict:
        predictions, references = [], []
        model.eval()
        pbar = tqdm(total=len(test_dataloader))
        for inputs, refs in test_dataloader:
            inputs = batch2gpu(inputs)
            with torch.no_grad():
                pred_tokens = model.generate(
                    **inputs, num_beams=4, max_length=args.max_target_length
                )
            preds = tokenizer.batch_decode(
                pred_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True
            )
            predictions += preds
            references += refs
            pbar.update()
        pbar.close()
        _predictions, _references = predictions, references
        if args.lang == "en":

            def preprocess(strs):
                return [word_tokenize(str) for str in strs]

            _predictions, _references = preprocess(predictions), preprocess(references)
        uni_F1_score(_predictions, _references)
        with open(args.predictions, "w") as f:
            f.write("\n".join(predictions))


if __name__ == "__main__":
    _, arg_name = sys.argv
    args = importlib.import_module(f"config.{arg_name}").Arguments()
    try:
        args.max_steps
    except AttributeError as e:
        print(str(e))
        args.max_steps = None
    try:
        args.posterior
    except AttributeError as e:
        print(str(e))
        args.posterior = False
    main(args)
