from io import StringIO

import pandas as pd
import json

import rag


# Script for creating document-level based Bosch AnnoCTR dataset

DATASET = "dev"
all_reports = []


df_raw = pd.read_json("D:/Datasets/cyber_threat_intelligence/dataset/anno-ctr-lrec-coling-2024/AnnoCTR/linking_mitre_only/" + DATASET + ".jsonl", lines=True)
df_report_list = df_raw.groupby('document')

mitre_table = rag.read_rag_db("../../database/rag_db.csv")

# Combine dev + train Dataset
# df_old = pd.read_json('../cti_datasets/bosch/bosch_cti_train_ds.json', lines=True)
# for old in df_old.values[0]:
#     all_reports.append(old)


i = 0
for _, cti in df_report_list:
    report = ""
    cti_id_label = []
    anno_name_label = []
    cti_entity_class = []
    i += 1
    print(i)

    for index, row in cti.iterrows():
        if report == "":
            f = open("D:/Datasets/cyber_threat_intelligence/dataset/anno-ctr-lrec-coling-2024/AnnoCTR/text/" + DATASET + "/" + row["document"] + ".txt", "r", encoding="utf8")
            report = f.read()

        label = row["label_link"].split("/")[-1]
        if label.startswith("T") or label.startswith("S") or label.startswith("M") or label.startswith("G"):
            cti_id_label.append(label)


        elif row["label_title"] != "No Annotation":
            anno_name_label.append((row["entity_type"], row["label_title"]))
        else:
            continue

        cti_entity_class.append(row["entity_class"])

    all_reports.append({"cti_report": report, "mitre_ids": cti_id_label, "anno_name_label": anno_name_label, "cti_entity_class": cti_entity_class})


with open('../cti_datasets/bosch/bosch_cti_' + DATASET + '_ds.json', 'w') as f:
    json.dump(all_reports, f)
