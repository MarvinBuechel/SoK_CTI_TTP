from itertools import chain

import pandas as pd
import json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import mitre
import ollama_api
import rag
from database import cti_reports

import finetuning.test_helper as th

import random


# Create Instruction LLM training set


#df = pd.read_json("../cti_datasets/bosch/bosch_cti_sentence_based_devtrain_ds.json")
#df = pd.read_json("../cti_datasets/tram/official_sok_split/train_split.json")
with open("../cti_datasets/tram/train_syn_sentence_list.json", "r") as file:
    data = json.load(file)

# get number of documents in tram dataset
#df["labels"] = df["labels"].apply(lambda x: x[0] if len(x) else "")
#print(df.groupby('labels').size())

mitre_table = rag.read_rag_db("../../database/rag_db.csv")
#df = pd.read_json("../cti_datasets/bosch/bosch_cti_traindev_ds.json")

# Concat dessertlab
#df2 = pd.read_csv("../cti_datasets/dessertlab/dataset.csv")
#df2.rename(columns={"label_tec": "labels"}, inplace=True)
#print(df2.groupby('labels').size())
#df2['labels'] = df2['labels'].apply(lambda x: [x])
#rows = df[df["labels"].str.len() == 0]
#df_all = pd.concat([df, df2])

#system = "You are a MITRE ATT&CK framework Cyber Security Expert."
system = "You are a Cyber Security Expert who finds MITRE ATT&CK framework concepts in CTI reports."




single_technique_pool = [
    "Based on the provided MITRE ATT&CK techniques and the given sentence, I extracted the following technique:",
    "Based on the provided MITRE ATT&CK techniques, I found that the following technique is present in the sentence:",
    "Based on the provided MITRE ATT&CK techniques, I found the following technique that matches the sentence:",
    "The MITRE ATT&CK technique used in the sentence is:",
    "Based on the provided MITRE ATT&CK techniques, I found that the following technique is contained within the sentence surrounded by grave accents (`):",
    "Based on the provided MITRE ATT&CK techniques, I extracted the following technique(s) that match the sentence:",
    "Based on the provided MITRE ATT&CK techniques and sentence, I extracted the following technique:",
]

multi_technique_pool = [
    "Based on the provided sentence, I extracted the following MITRE ATT&CK techniques:",
    "The extracted MITRE ATT&CK techniques are:",
    "MITRE ATT&CK techniques that are in the sentence:",
    "MITRE ATT&CK techniques in this sentence:",
    "Based on the provided MITRE ATT&CK techniques and the sentence:",
]

none_technique_pool = [
    "I extracted the following MITRE ATT&CK techniques:\n\n* None",
    "I extracted the following MITRE ATT&CK techniques from the sentence:\n\n* None",
    "There are no MITRE ATT&CK techniques mentioned in the sentence.",
    "Based on the provided MITRE ATT&CK techniques and sentence distance metrics, I found that there are no MITRE ATT&CK techniques mentioned in the given sentence surrounded by grave accents (`).",
    "The given sentence does not appear to be related to any of the MITRE ATT&CK techniques listed.",
    "None. There are no MITRE ATT&CK techniques mentioned in the sentence.",
    "There are no MITRE ATT&CK techniques mentioned in the sentence surrounded by grave accents (`).",
    "No MITRE ATT&CK techniques are surrounded by grave accents (`) in the given sentence.",
    "I extracted the MITRE ATT&CK techniques mentioned in the sentence:\n\n* None (there are no MITRE ATT&CK techniques mentioned in the sentence)",
    "There are no MITRE ATT&CK techniques in the sentence.",
    "None. There are no MITRE ATT&CK techniques mentioned in the sentence surrounded by grave accents (`). The provided sentences are descriptions of MITRE ATT&CK techniques, not actual techniques used in the sentence.",
    "There are no MITRE ATT&CK techniques in the sentence surrounded by grave accents (`). The sentence appears to be unrelated to any of the described techniques.",
]


instruction_list = []
ollama_api.ollama_init()

#tram
#sentences = df["cti_report"]
#labels = df["label"]
#bosch
#sentences = [df["sentence"]]
#labels = [df["labels"]]
#tram augmented
sentences = []
labels = []
for key, entries in data.items():
    for entry in entries:
        sentences.append(entry.get("augmented_sentence", ""))
        labels.append(entry.get("labels", []))
sentences = [sentences]
labels = [labels]

for sentence_list, label_list in tqdm(zip(sentences, labels)):
    #sentence = cti_preprocessing.replace_iocs(sentence)

    for sentence, label in tqdm(zip(sentence_list, label_list)):
        techniques = []

        for id in label:
            #id = id.split(".")[0] # only techniques, no subtechniques
            concept = mitre_table.loc[mitre_table['ID'] == id]
            if len(concept) == 0:
                print("Failure ID: " + id)
                continue
            if id.startswith("T"):
                techniques.append("* " + id + ": " + concept["name"].values[0])


        user_query = mitre.create_mitre_sentence_based_prompt(sentence)

        if len(techniques) > 0:
            if len(techniques) == 1:
                response = random.choice(single_technique_pool) + "\n"
            else:
                response = random.choice(multi_technique_pool) + "\n"
            for i, tech in enumerate(techniques):
                response += tech + "\n"
            response = response[:-1]  # remove last \n
        else:
            response = random.choice(none_technique_pool)

        instruction_list.append(
            {"messages": [{"role": "system", "content": system},
                          {"role": "user", "content": user_query},
                          {"role": "assistant", "content": response}]}
        )


with open('../cti_datasets/tram/sentence_based_augmented_train_instructs.json', 'w') as f:
    json.dump(instruction_list, f)
