import openai
import time
import socket
from tqdm import tqdm

api_keys = []
api_key_idx = 0
few_shot_num = 8

with open("wow_test.txt", "r", encoding="utf-8") as f:
    data = f.readlines()

with open("fs_examples/woi_" + str(few_shot_num) + ".txt", "r", encoding="utf-8") as f:
    examples = "".join(f.readlines())

start_idx = 0
end_idx = 3556

for i, l in tqdm(enumerate(data[start_idx:end_idx])):
    prompt = l.rstrip()
    resText = "hhh"
    instruction = (
        "Generate a query according to the dialogue history:\n"
        + examples.lower()
        + r"[history]:\n"
        + prompt.lower()
        + r"\n[query]:\n"
    )
    print(instruction)
    # break

    try:
        openai.api_key = api_keys[api_key_idx]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": instruction},
            ],
        )
    except openai.error.ServiceUnavailableError as e:
        print(str(e))
        time.sleep(6)
        api_key_idx = (api_key_idx + 1) % len(api_keys)
        openai.api_key = api_keys[api_key_idx]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": instruction},
            ],
        )
    except openai.error.RateLimitError as e:
        print(str(e))
        if "quota" in str(e):
            time.sleep(6)
            _api_key = openai.api_key
            print(_api_key, "is N/A")
            api_key_idx = (api_key_idx + 1) % len(api_keys)
            openai.api_key = api_keys[api_key_idx]
            api_keys.remove(_api_key)
            print("current API keys:", api_keys)
            api_key_idx = api_keys.index(openai.api_key)
            i -= 1
            continue
        else:
            api_key_idx = (api_key_idx + 1) % len(api_keys)
            openai.api_key = api_keys[api_key_idx]
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": instruction},
                ],
            )
    except openai.error.APIError as e:
        print(str(e))
        time.sleep(6)
        api_key_idx = (api_key_idx + 1) % len(api_keys)
        openai.api_key = api_keys[api_key_idx]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": instruction},
            ],
        )
    except socket.timeout as e:
        print(str(e))
        time.sleep(6)
        api_key_idx = (api_key_idx + 1) % len(api_keys)
        openai.api_key = api_keys[api_key_idx]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": instruction},
            ],
        )
    except openai.error.Timeout as e:
        print(str(e))
        time.sleep(6)
        api_key_idx = (api_key_idx + 1) % len(api_keys)
        openai.api_key = api_keys[api_key_idx]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": instruction},
            ],
        )
    except openai.error.AuthenticationError as e:
        print(str(e))
        _api_key = openai.api_key
        print(_api_key, "is N/A")
        api_key_idx = (api_key_idx + 1) % len(api_keys)
        openai.api_key = api_keys[api_key_idx]
        api_keys.remove(_api_key)
        print("current API keys:", api_keys)
        api_key_idx = api_keys.index(openai.api_key)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": instruction},
            ],
        )
    resText = response.choices[0].message.content
    print(str(i + start_idx) + "\t" + prompt + "\t" + resText + "\n")
    with open(
        "wow_test_fs"
        + str(few_shot_num)
        + "_answer_"
        + "{0:04d}".format(start_idx)
        + ".txt",
        "a",
        encoding="utf-8",
    ) as fo:
        fo.write(str(i + start_idx) + "\t" + prompt + "\t" + resText + "\n")
    time.sleep(6)
    api_key_idx = (api_key_idx + 1) % len(api_keys)
