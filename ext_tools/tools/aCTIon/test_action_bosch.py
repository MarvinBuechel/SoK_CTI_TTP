import pandas as pd
from itertools import chain
from tqdm import tqdm
import main as action
from collections import defaultdict
from metrics import calculate_metrics

from const import BOSCH_TECHNIQUES_LABELS, BOSCH_TECHNIQUES_10_LABELS, BOSCH_TECHNIQUES_25_LABELS, BOSCH_TECHNIQUES_50_LABELS


label_sets = {
    "10": BOSCH_TECHNIQUES_10_LABELS, 
    "25": BOSCH_TECHNIQUES_25_LABELS, 
    "50": BOSCH_TECHNIQUES_50_LABELS,
    "118": BOSCH_TECHNIQUES_LABELS
}


data = pd.read_json("../../dataset/bosch_test.json")
test_docs = data.groupby('document').agg({
    'labels': lambda x: list(set(chain.from_iterable(x))),
    'sentence': lambda x: "\n".join(list(x))
}).reset_index()


f1 = defaultdict(lambda: 0.0)
prec = defaultdict(lambda: 0.0)
rec = defaultdict(lambda: 0.0)

f1_all = 0.0
prec_all = 0.0
rec_all = 0.0

for i in tqdm(range(len(test_docs))):
    pred = action.main(test_docs.iloc[i].sentence)
    true_labels = test_docs.iloc[i].labels
    true_labels = [x for x in true_labels if x in BOSCH_TECHNIQUES_LABELS] # we only want technique labels

    f1_all += calculate_metrics(pred, true_labels, 'f1')
    prec_all += calculate_metrics(pred, true_labels, 'precision')
    rec_all += calculate_metrics(pred, true_labels, 'recall')

    for lset, labels in label_sets.items():
        pred_filtered = [x for x in pred if x in labels]  # filter predictions
        true_labels_filt = [x for x in test_docs.iloc[i].labels if x in labels]
        f1[lset] += calculate_metrics(pred_filtered, true_labels_filt, 'f1')
        prec[lset] += calculate_metrics(pred_filtered, true_labels_filt, 'precision')
        rec[lset] += calculate_metrics(pred_filtered, true_labels_filt, 'recall')

for lset, labels in label_sets.items():
    f1[lset] = round(f1[lset] / len(test_docs)*100, 2)
    prec[lset] = round(prec[lset] / len(test_docs)*100, 2)
    rec[lset] = round(rec[lset] / len(test_docs)*100, 2)

f1_all = round(f1_all / len(test_docs) * 100, 2)
prec_all = round(prec_all / len(test_docs) * 100, 2)
rec_all = round(rec_all / len(test_docs) * 100, 2)


with open("bosch_scores.txt", "w") as f:
    f.write(f"F1 {[lset for lset in f1]}: {[f1[lset] for lset in f1]}\n")
    f.write(f"PREC {[lset for lset in f1]}: {[prec[lset] for lset in prec]}\n")
    f.write(f"REC {[lset for lset in f1]}: {[rec[lset] for lset in rec]}\n")
    f.write(f"F1 (all): {f1_all}\n")
    f.write(f"PREC (all): {prec_all}\n")
    f.write(f"REC (all): {rec_all}\n")
