import json
import os

import pandas as pd
from tqdm import tqdm
import ollama_api


# Create data augmentation for TRAM2 dataset

def get_dataset_sentence(path):
    df = pd.read_json(path)

    reports = df["cti_report"]
    mitre_ids = df["label"]

    return zip(reports, mitre_ids)


dataset = []

dataset.extend(get_dataset_sentence("../cti_datasets/tram/official_sok_split/train_split.json"))

syn_sentence = {}
JSON_PATH = "finetuning/cti_datasets/tram/syn_sentence_list.json"


if os.path.exists(JSON_PATH):
    try:
        with open(JSON_PATH) as json_file:
            syn_sentence = json.load(json_file)
    except Exception as e:
        print(f"Fail: {e}")

i = 0

while True:
    i = i + 1

    for data in tqdm(dataset):
        text_list, label_list = data

        for text, label in zip(text_list, label_list):

            print(str(i) + " ")

            # ######### Prompt ########
            prompt = "Rephrase the following sentence in the brackets without semantically changing the content: \"" + text + "\"\n\nThe following sentences have already been generated from this original sentence, create a semantically identical sentence that also differs syntactically from the following sentences, do not output anything else:\n"
            if text in syn_sentence:
                for aug_sentence in syn_sentence[text]:
                    prompt += "-" + aug_sentence["augmented_sentence"] + "\n"

            # ######### LLM ###########
            response = ollama_api.query_llm_chat(prompt, model="llama3.3:70b", context_size=8192, temperature=1.0)

            if len(response) > 0:
                if response[0] == "\"":
                    response = response[1:]
                elif response[-1] == "\"":
                    response = response[:-1]

            if text not in syn_sentence:
                syn_sentence[text] = []
            syn_sentence[text].append({"augmented_sentence": response, "labels": label})

        with open(JSON_PATH, "w") as json_file:
            json.dump(syn_sentence, json_file, indent=4)
