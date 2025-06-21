
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, classification_report
from collections import defaultdict
import numpy as np
import pandas as pd


def calculate_metrics(row, metric: str):
    # unique all ids
    unique_prediction_list = list(set(row['pred']))
    unique_true_label_list = list(set(row['true']))

    if len(unique_prediction_list) == 0 and len(unique_true_label_list) == 0:  # if empty labels are correct
        return 1.

    # Initialize variables for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Iterate through each item in the prediction list
    for item in unique_prediction_list:
        # Check if the item is in the true label list
        if item in unique_true_label_list:
            # If the item is in both lists, it's a true positive
            true_positives += 1
        else:
            # If the item is not in the true label list, it's a false positive
            false_positives += 1

    # Calculate false negatives
    false_negatives = len(unique_true_label_list) - true_positives

    # Calculate Jaccard Index or Critical Success Index
    accuracy = true_positives / (true_positives + false_positives + false_negatives)

    # Calculate precision
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0

    # Calculate F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0

    if metric == 'f1':
        return f1
    if metric == 'accuracy':
        return accuracy
    if metric == 'precision':
        return precision
    if metric == 'recall':
        return recall


def calc_results_per_document(results, labels_map=None):
    # we will save the results inside here
    out = defaultdict(list)
    for doc in results:
        true = results[doc]["labels"]
        predictions = results[doc]["preds"]
        all_true_ttps = []
        all_pred_ttps = []
        
        if labels_map:
            for true_values, prediction_values in zip(true, predictions):
                true_i, = np.where(true_values == 1)
                pred_i, = np.where(prediction_values == 1)
                true_ttps = [labels_map[i] for i in true_i]
                pred_ttps = [labels_map[i] for i in pred_i]
                all_true_ttps.extend(true_ttps)
                all_pred_ttps.extend(pred_ttps)
        else:
            all_true_ttps.extend(true)
            all_pred_ttps.extend(predictions)

        out['doc_title'].append(doc)
        out['true'].append(all_true_ttps)
        out['pred'].append(all_pred_ttps)

    out_df = pd.DataFrame(out)
    out_df["precision"] = out_df.apply(lambda x: calculate_metrics(x, "precision"), axis=1)
    out_df["recall"] = out_df.apply(lambda x: calculate_metrics(x, "recall"), axis=1)
    out_df["accuracy"] = out_df.apply(lambda x: calculate_metrics(x, "accuracy"), axis=1)
    out_df["f1"] = out_df.apply(lambda x: calculate_metrics(x, "f1"), axis=1)
    return out_df


def calc_results_per_sentence(results):
    out = defaultdict(list)
    true = results['labels']
    pred = results['preds']
    
    micro_f1 = f1_score(true, pred, average="micro", zero_division=0.0)
    macro_f1 = f1_score(true, pred, average="macro", zero_division=0.0)
    samples_f1 = f1_score(true, pred, average="samples", zero_division=0.0)
    weighted_f1 = f1_score(true, pred, average="weighted", zero_division=0.0)
    accuracy = accuracy_score(true, pred)
    micro_precision = precision_score(true, pred, average="micro", zero_division=0.0)
    macro_precision = precision_score(true, pred, average="macro", zero_division=0.0)
    samples_precision = precision_score(true, pred, average="samples", zero_division=0.0)
    weighted_precision = precision_score(true, pred, average="weighted", zero_division=0.0)
    micro_recall = recall_score(true, pred, average="micro", zero_division=0.0)
    macro_recall = recall_score(true, pred, average="macro", zero_division=0.0)
    samples_recall = recall_score(true, pred, average="samples", zero_division=0.0)
    weighted_recall = recall_score(true, pred, average="weighted", zero_division=0.0)

    out['doc_title'].append("per_sentence")
    out['micro_f1'].append(micro_f1)
    out['macro_f1'].append(macro_f1)
    out['samples_f1'].append(samples_f1)
    out['weighted_f1'].append(weighted_f1)
    out['micro_precision'].append(micro_precision)
    out['macro_precision'].append(macro_precision)
    out['samples_precision'].append(samples_precision)
    out['weighted_precision'].append(weighted_precision)
    out['micro_recall'].append(micro_recall)
    out['macro_recall'].append(macro_recall)
    out['samples_recall'].append(samples_recall)
    out['weighted_recall'].append(weighted_recall)
    out['accuracy'].append(accuracy)
    out_df = pd.DataFrame(out)
    return out_df
