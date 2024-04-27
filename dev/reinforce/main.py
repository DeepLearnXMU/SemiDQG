import importlib
import sys
import math
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer, set_seed
from torch.utils.data import DataLoader
from dataset import RAQGDataset, QueryGenDataset
from utils import *
from nltk import word_tokenize
from model import T5ForConditionalGenerationRAQG


def main(args):
    print(args.__dict__)
    # Set random seed
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, truncation_side="left"
    )
    model = T5ForConditionalGenerationRAQG.from_pretrained(args.model_name_or_path)
    model = model.cuda()

    def collate_fn(batch):
        inputs, targets, scores = zip(*batch)
        inputs, targets, scores = (
            inputs[0],
            targets[0],
            scores[0],
        )  # for batch size is 1
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=args.max_target_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            labels["input_ids"][labels["input_ids"] == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels["input_ids"]
        # prepare decoder_input_ids
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(
            labels=model_inputs["labels"]
        )
        model_inputs["decoder_input_ids"] = decoder_input_ids
        model_inputs["rewards"] = torch.tensor(scores)

        return model_inputs

    def eval_collate_fn(batch):
        inputs, targets = zip(*batch)
        inputs, targets = list(inputs), list(targets)
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return model_inputs, targets

    train_dataset = RAQGDataset(
        args.train_file,
        args.topk,
        args.threshold,
        args.posterior,
        args.beam_top_filter,
        args.score_normalization,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    valid_dataset = QueryGenDataset(args.valid_file, args.posterior)
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.eval_batch_size if hasattr(args, "eval_batch_size") else 16,
        shuffle=False,
        collate_fn=eval_collate_fn,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    def evaluate(model):
        predictions, references = [], []
        model.eval()
        for inputs, refs in tqdm(valid_dataloader):
            inputs = preprocess_batch(inputs)
            with torch.no_grad():
                pred_tokens = model.generate(
                    **inputs, num_beams=4, max_length=args.max_target_length
                )
            preds = tokenizer.batch_decode(
                pred_tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True
            )
            predictions += preds
            references += refs
        if args.lang == "en":

            def preprocess(strs):
                return [word_tokenize(str) for str in strs]

            predictions, references = preprocess(predictions), preprocess(references)
        return uni_F1_score(predictions, references)

    # eval_score = evaluate(model)
    # print(f"Step {0}, Evaluation score {round(eval_score, 4)}")

    if args.do_train:
        backward_steps, global_steps = 0, 0
        train_loss, support_loss = 0.0, 0.0
        model.train()
        if args.max_steps:
            if args.save_model and args.save_steps is None:
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
        for epoch in range(math.ceil(epoches)):
            model.train()
            for batch in train_dataloader:
                batch = preprocess_batch(batch)
                loss = model.forward_ra(
                    **batch,
                    use_smooth_len_norm=args.use_smooth_len_norm,
                    no_len_norm=args.no_len_norm,
                )
                loss /= args.gradient_accumulation_steps
                train_loss += loss.item()
                loss.backward()
                backward_steps += 1
                if backward_steps % args.gradient_accumulation_steps == 0:
                    pbar.update(1)
                    pbar.set_description(
                        f"Epoch {epoch+1} Loss {round(train_loss, 3)} "
                        f"Support-Loss {round(support_loss, 3)} "
                    )
                    global_steps += 1
                    train_loss = 0.0
                    optimizer.step()
                    optimizer.zero_grad()
                    if args.max_steps and global_steps % args.save_steps == 0:
                        eval_score = evaluate(model)
                        print(
                            f"Step {global_steps}, Evaluation score {round(eval_score, 4)}"
                        )
                        if args.save_model:
                            save_model(args, model, tokenizer, step=global_steps)
                            print(f"saving model at checkpoint step {global_steps}")
                        model.train()
                    if global_steps == args.max_steps:
                        break
            if args.max_steps is None:
                eval_score = evaluate(model)
                print(f"Step {global_steps}, Evaluation score {round(eval_score, 4)}")
                if args.save_model:
                    save_model(args, model, tokenizer, step=global_steps)
                    print(f"saving model at checkpoint step {global_steps}")
                model.train()

        save_model(args, model, tokenizer, step=global_steps)
        print(f"saving model at checkpoint step {global_steps}")


if __name__ == "__main__":
    _, arg_name = sys.argv
    args = importlib.import_module(f"config.{arg_name}").Arguments()
    if not hasattr(args, "beam_top_filter"):
        args.beam_top_filter = False
    if not hasattr(args, "max_steps"):
        args.max_steps = None
    if not hasattr(args, "use_smooth_len_norm"):
        args.use_smooth_len_norm = True
    if not hasattr(args, "no_len_norm"):
        args.no_len_norm = False
    if not hasattr(args, "score_normalization"):
        args.score_normalization = True
    main(args)
