#
import pickle

import sys

if len(sys.argv) < 2:
    print("Usage: python unlabeled_nvidia_embed.py <device>")
    print("device should be 'cuda', 'cuda:0', 'cpu', etc.")
    sys.exit(1)

device = sys.argv[1]
print("using device: %s" % device)

print("loading embeddings from 'datasets' folder...")
#
with open("datasets/mitre_embeddings.pickle", "rb") as f:
    mitre_embeddings_w_id = pickle.load(f)

with open("datasets/nvidia-mitre-embeddings.pickle", "rb") as f:
    mitre_embeddings = pickle.load(f)

with open("datasets/nvidia-tram-train-embeddings.pickle", "rb") as f:
    tram_train_embeddings = pickle.load(f)

with open("datasets/nvidia-tram-test-embeddings.pickle", "rb") as f:
    tram_test_embeddings = pickle.load(f)

with open("datasets/nvidia-bosch-train-embeddings.pickle", "rb") as f:
    bosch_train_embeddings = pickle.load(f)

with open("datasets/nvidia-bosch-test-embeddings.pickle", "rb") as f:
    bosch_test_embeddings = pickle.load(f)

print("converting mitre embeddings to dict with ttp ids as keys...")
#
ttp_ids = list(mitre_embeddings_w_id.keys())
i = 0
mitre_embeddings_fix = {}
for batch in mitre_embeddings:
    for emb in batch:
        try:
            mitre_embeddings_fix[ttp_ids[i]] = emb
        except:
            print("can't find ttp id %s" % i)
        i+=1

#
import torch.nn.functional as F

tram_train_embeddings = {k.item(): F.normalize(v.to(device), p=2, dim=0) for k,v in tram_train_embeddings.items()}
tram_test_embeddings = {k: F.normalize(v.to(device), p=2, dim=0) for k,v in tram_test_embeddings.items()}

bosch_train_embeddings = {k.item(): F.normalize(v.to(device), p=2, dim=0) for k,v in bosch_train_embeddings.items()}
bosch_test_embeddings = {k: F.normalize(v.to(device), p=2, dim=0) for k,v in bosch_test_embeddings.items()}

mitre_embeddings_fix = {k: v.to(device) for k,v in mitre_embeddings_fix.items()}

#
import torch

#
ttp_embeddings = torch.vstack(list(mitre_embeddings_fix.values()))
ttp_embeddings = F.normalize(ttp_embeddings, p=2, dim=1)
ttp_embeddings.to(device)

#
from const import TRAM_TECHNIQUES_LABELS, BOSCH_TECHNIQUES_LABELS

tram_bosch = sorted(list(set(BOSCH_TECHNIQUES_LABELS).union(TRAM_TECHNIQUES_LABELS)))

ttp_ids_bosch = [i for i,t in enumerate(ttp_ids) if t in BOSCH_TECHNIQUES_LABELS]
ttp_ids_tram = [i for i,t in enumerate(ttp_ids) if t in TRAM_TECHNIQUES_LABELS]
ttp_ids_all = [i for i,t in enumerate(ttp_ids)]

# [markdown]
# ## TRAM

#
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

tram_results = {}

ids_per_label_set = {
    "tram_t": ttp_ids_tram,
    "bosch_t": ttp_ids_bosch,
    "all_mitre": ttp_ids_all # this contains ids
}

labels_per_label_set = {
    "tram_t": TRAM_TECHNIQUES_LABELS,
    "bosch_t": BOSCH_TECHNIQUES_LABELS,
    "all_mitre": ttp_ids # this contains labels
}

df = pd.read_json("datasets/tram_train.json")

print("tuning thresholds for TRAM...")
for label_set in tqdm(["tram_t", "all_mitre"]):
    
    labels = labels_per_label_set[label_set]
    label_ids = ids_per_label_set[label_set]

    lb = MultiLabelBinarizer()
    lb.fit([labels])

    best_f1 = -1
    best_tau = 0.5

    for tau in np.linspace(0.25, 0.75, num=20):

        all_labels = []
        all_preds = []

        for i, emb in tram_train_embeddings.items():

            # Compute similarities
            similarities = emb @ ttp_embeddings.T  # Resulting shape: (number_of_embeddings,)
            
            # Create a mask based on ttp_ids_bosch_all
            mask = torch.zeros(similarities.shape, dtype=torch.bool)  # Match the shape of similarities
            mask[label_ids] = True  # Assume ttp_ids_bosch_all contains valid indices

            # Set values not in the mask to 0
            similarities[~mask] = 0  # Directly apply the mask to the 1D tensor
            indices_above_threshold = torch.where(similarities > tau)[0]
            ttp_values = [ttp_ids[i] for i in indices_above_threshold]
            tram_labels = df.iloc[i].labels
            tram_labels = [l for l in tram_labels if l in labels]
            all_labels.append(tram_labels)
            all_preds.append(ttp_values)

        y_true = lb.transform(all_labels)
        y_pred = lb.transform(all_preds)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
        
    tram_results[label_set] = {
        "best_f1": best_f1,
        "best_tau": best_tau
    }
    
    print("label_set=%s best_f1=%s best_tau=%s" % (label_set, best_f1, best_tau))

# [markdown]
# ## AnnoCTR

#
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


bosch_results = {}

df = pd.read_json("datasets/bosch_train.json")

print("tuning thresholds for AnnoCTR...")
for label_set in tqdm(["bosch_t", "all_mitre"]):
    
    labels = labels_per_label_set[label_set]
    label_ids = ids_per_label_set[label_set]

    lb = MultiLabelBinarizer()
    lb.fit([labels])

    best_f1 = -1
    best_tau = 0.5

    for tau in np.linspace(0.25, 0.75, num=20):

        all_labels = []
        all_preds = []

        for i, emb in bosch_train_embeddings.items():

            # Compute similarities
            similarities = emb @ ttp_embeddings.T  # Resulting shape: (number_of_embeddings,)
            
            # Create a mask based on ttp_ids_bosch_all
            mask = torch.zeros(similarities.shape, dtype=torch.bool)  # Match the shape of similarities
            mask[label_ids] = True  # Assume ttp_ids_bosch_all contains valid indices

            # Set values not in the mask to 0
            similarities[~mask] = 0  # Directly apply the mask to the 1D tensor
            indices_above_threshold = torch.where(similarities > tau)[0]
            ttp_values = [ttp_ids[i] for i in indices_above_threshold]
            bosch_labels = df.iloc[i].labels
            bosch_labels = [l for l in bosch_labels if l in labels]
            all_labels.append(bosch_labels)
            all_preds.append(ttp_values)

        y_true = lb.transform(all_labels)
        y_pred = lb.transform(all_preds)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
        
    bosch_results[label_set] = {
        "best_f1": best_f1,
        "best_tau": best_tau
    }
    
    print("label_set=%s best_f1=%s best_tau=%s" % (label_set, best_f1, best_tau))


#
# tram_results = {'tram_t': {'best_f1': np.float64(0.172825459730935),
#   'best_tau': np.float64(0.48684210526315785)},
#  'all_mitre': {'best_f1': np.float64(0.17282545973093494),
#   'best_tau': np.float64(0.48684210526315785)}}

# bosch_results = {'bosch_t': {'best_f1': np.float64(0.18945444005115686),
#   'best_tau': np.float64(0.4342105263157895)},
#  'all_mitre': {'best_f1': np.float64(0.18967108320730455),
#   'best_tau': np.float64(0.4342105263157895)}}

#
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

tram_out = {}

df = pd.read_json("datasets/tram_test.json")

print("predicting labels for TRAM...")
for label_set in tqdm(["tram_t", "all_mitre"]):
    
    labels = labels_per_label_set[label_set]
    label_ids = ids_per_label_set[label_set]

    all_labels = []
    all_preds = []

    lb = MultiLabelBinarizer()
    lb.fit([labels])

    tau = tram_results[label_set]["best_tau"]
    tram_out[label_set] = {}

    for doc_name, df_doc in df.groupby("doc_title"):
        all_labels = []
        all_preds = []

        for idx, row in df_doc.iterrows():

            tram_labels = [l for l in row.labels if l in labels]
            emb = tram_test_embeddings[list(tram_test_embeddings.keys())[idx]].to(device)
            # Compute similarities
            similarities = emb @ ttp_embeddings.T  # Resulting shape: (number_of_embeddings,)
            
            # Create a mask based on ttp_ids_bosch_all
            mask = torch.zeros(similarities.shape, dtype=torch.bool)  # Match the shape of similarities
            mask[label_ids] = True  # Assume ttp_ids_bosch_all contains valid indices

            # Set values not in the mask to 0
            similarities[~mask] = 0  # Directly apply the mask to the 1D tensor
            indices_above_threshold = torch.where(similarities > tau)[0]
            ttp_values = [ttp_ids[i] for i in indices_above_threshold]

            all_labels.extend(tram_labels)
            all_preds.extend(ttp_values)

        tram_out[label_set][doc_name] = {
            "labels": list(set(all_labels)),
            "preds": list(set(all_preds))
        }

#
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

bosch_out = {}

df = pd.read_json("datasets/bosch_test.json")

print("predicting labels for AnnoCTR...")
for label_set in tqdm(["bosch_t", "all_mitre"]):
    
    labels = labels_per_label_set[label_set]
    label_ids = ids_per_label_set[label_set]

    all_labels = []
    all_preds = []

    lb = MultiLabelBinarizer()
    lb.fit([labels])

    tau = bosch_results[label_set]["best_tau"]
    bosch_out[label_set] = {}

    for doc_name, df_doc in df.groupby("document"):
        all_labels = []
        all_preds = []

        for idx, row in df_doc.iterrows():

            bosch_labels = [l for l in row.labels if l in labels]

            emb = bosch_test_embeddings[idx]
            # Compute similarities
            similarities = emb @ ttp_embeddings.T  # Resulting shape: (number_of_embeddings,)
            
            # Create a mask based on ttp_ids_bosch_all
            mask = torch.zeros(similarities.shape, dtype=torch.bool)  # Match the shape of similarities
            mask[label_ids] = True  # Assume ttp_ids_bosch_all contains valid indices

            # Set values not in the mask to 0
            similarities[~mask] = 0  # Directly apply the mask to the 1D tensor
            indices_above_threshold = torch.where(similarities > tau)[0]
            ttp_values = [ttp_ids[i] for i in indices_above_threshold]

            all_labels.extend(bosch_labels)
            all_preds.extend(ttp_values)

        bosch_out[label_set][doc_name] = {
            "labels": list(set(all_labels)),
            "preds": list(set(all_preds))
        }

#
from test_common import calc_results_per_document

output = {
    "model_name": [],
    "dataset_name": [],
    "label_set": [],
    "f1": [],
    "accuracy": [],
    "precision": [],
    "recall": []
}

for dataset, out in zip(["tram", "bosch"], [tram_out, bosch_out]):
    for label_set in out:
        results_df = calc_results_per_document(out[label_set])
        f1 = results_df.f1.mean()
        accuracy = results_df.accuracy.mean()
        precision = results_df.precision.mean()
        recall = results_df.recall.mean()
        output["model_name"].append("nvidia-embed")
        output["dataset_name"].append(dataset)
        output["label_set"].append(label_set)
        output["f1"].append(f1)
        output["accuracy"].append(accuracy)
        output["precision"].append(precision)
        output["recall"].append(recall)

#
final_df = pd.DataFrame(output).sort_values(by="f1")
final_df.f1 = (final_df.f1 * 100).round(2)
final_df.precision = (final_df.precision * 100).round(2)
final_df.recall = (final_df.recall * 100).round(2)
final_df = final_df.sort_values(by=["dataset_name", "label_set"], ascending=[True, True])

print(final_df[["model_name", "dataset_name", "label_set", "f1", "precision", "recall"]])

