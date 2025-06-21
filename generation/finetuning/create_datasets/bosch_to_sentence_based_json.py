from io import StringIO

import pandas as pd
import json

import os


# Script for creating sentence based Bosch AnnoCTR dataset

DATASET = "train"
all_sentences = []

print(os.getcwd())

df_raw = pd.read_json(
    "D:/Datasets/cyber_threat_intelligence/dataset/anno-ctr-lrec-coling-2024/AnnoCTR/linking_mitre_only/" + DATASET + ".jsonl",
    lines=True)
df_report_list = df_raw.groupby('document')

#mitre_table = rag.read_rag_db("../../database/rag_db.csv")

# Combine dev + train Dataset
#df_old = pd.read_json('../cti_datasets/bosch/bosch_cti_sentence_based_train_ds.json', lines=True)
#for old in df_old.values[0]:
#    all_sentences.append(old)


i = 0
for _, cti in df_report_list:
    i += 1
    print(i)

    for index, row in cti.iterrows():
        cti_id_label = []
        sentence = ""

        label = row["label_link"].split("/")[-1]
        if label.startswith("T"):
            cti_id_label.append(label)

        sentence = row["_context_left"] + row["mention"] + row["_context_right"]

        #sentence = cti_preprocessing.replace_iocs(sentence)

        for i, saved_sentence in enumerate(all_sentences):
            if saved_sentence["sentence"] == sentence:
                all_sentences[i]["labels"].extend(cti_id_label)
                sentence = ""

        if sentence != "":
            all_sentences.append({"sentence": sentence, "labels": cti_id_label, "document": row["document"]})



with open('../cti_datasets/bosch/bosch_cti_sentence_based_' + DATASET + '_ds.json', 'w') as f:
    json.dump(all_sentences, f)
