import os
import re
import jsonlines
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenizer.add_tokens(
    ["[apprentice_persona]", "[dialog_history]", "[apprentice]", "[wizard]"]
)


def get_model_input(data):
    data = list(data.values())[0]
    dialog = [data["apprentice_persona"].strip()]
    model_input = []
    cur_queries = {}
    last_query = None
    for idx, turn in enumerate(data["dialog_history"]):
        action = turn["action"].strip()
        text = re.sub("\s+", " ", turn["text"].strip())
        if action == "Apprentice => Wizard":
            dialog.append(text)
        elif action == "Wizard => Apprentice":
            urls = {
                page["url"][:-1] if page["url"][-1] == "/" else page["url"]
                for page in turn["context"]["contents"]
            }
            if urls:
                select_state = False
                for state in turn["context"]["selected_contents"]:
                    for _state in state:
                        if _state:
                            select_state = True
                select_query = None
                for k, v in cur_queries.items():
                    if urls == v:
                        select_query = k
                        break
                if select_query and select_state:
                    model_input.append(
                        {
                            "dialogue": "\n".join(dialog),
                            "response": text,
                            "query": select_query,
                        }
                    )
            dialog.append(text)
            # cur_queries = {}
        elif action == "Wizard => SearchAgent":
            last_query = text
        elif action == "SearchAgent => Wizard":
            urls = {
                page["url"][:-1] if page["url"][-1] == "/" else page["url"]
                for page in turn["context"]["contents"]
            }
            cur_queries[last_query] = urls
        else:
            raise Exception("UNKNOWN ACTION")
    return model_input


for split in ["train", "valid", "test"]:
    data = []
    input_len, output_len = [], []
    with jsonlines.open(
        f"../saved_data/wizard_of_interent/{split}.jsonl", "r"
    ) as reader:
        num_dialogues = 0
        for line in reader:
            num_dialogues += 1
            model_inputs = get_model_input(line)
            data += model_inputs
            input_len.append(len(tokenizer.tokenize(data[-1]["dialogue"])))
    output_len += [len(tokenizer.tokenize(dialog["query"])) for dialog in data]
    print(
        len(data),
        num_dialogues,
        np.mean(input_len),
        max(input_len),
        np.mean(output_len),
        max(output_len),
    )
    output_file = f"../saved_data/data_woi/{split}.json"
    if not os.path.exists("../saved_data/data_woi"):
        os.makedirs("../saved_data/data_woi")
    with jsonlines.open(output_file, "w") as writer:
        for x in data:
            writer.write(x)
