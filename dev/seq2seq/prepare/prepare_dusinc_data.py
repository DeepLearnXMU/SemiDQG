import os
import re
import jsonlines
import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")
tokenizer.add_tokens(
    ["[apprentice_persona]", "[dialog_history]", "[apprentice]", "[wizard]"]
)

"""
Source format:
	- coversation: List
		- role
		- utterance
		- use_kg_label (bot): bool
		- use_knowledge (bot)
		- use_query (bot) -> query
		- other_search (bot) (ignore temporaly)
			- search_query
			- search_knowledge
	- user_topical: List
	- user_location: str

Target format:
	- dialogue
	- response
	- query
"""


def get_model_input(line):
    model_input = []
    dialogue = (
        "\n".join([re.sub("\s+", "", l.strip()) for l in line["user_topical"]])
        + "\n"
        + re.sub("\s+", "", line["user_location"].strip())
    )
    for item in line["conversation"]:
        response = (
            re.sub("\s+", "", item["utterance"].strip()) if "utterance" in item else ""
        )
        if response and "use_query" in item and item["use_query"]:
            model_input.append(
                {
                    "dialogue": dialogue,
                    "response": response,
                    "query": re.sub("\s+", "", item["use_query"].strip()),
                }
            )
        if dialogue:
            dialogue += "\n"
        dialogue += response
    return model_input


for split in ["train", "dev", "test_dial_1", "test_query_1"]:
    data = []
    input_len, output_len = [], []
    with jsonlines.open(f"../saved_data/DuSinc/{split}.txt", "r") as reader:
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
    if split == "dev":
        split = "valid"
    output_file = f"../saved_data/data_dusinc/{split}.json"
    if not os.path.exists("../saved_data/data_dusinc"):
        os.makedirs("../saved_data/data_dusinc")
    with jsonlines.open(output_file, "w") as writer:
        for x in data:
            writer.write(x)
