import re
from typing import List

import pandas as pd

mitre_id_pattern = r"(M\d{4}|G\d{4}|S\d{4}|T\d{4}(?:\.\d+)?|TA\d{4})"  # get all MITRE IDs


# Extract all valid MITRE IDs based on regex pattern
def extract_mitre_ids(mitre: pd.DataFrame, text):
    # Extract all MITRE IDs based on regex pattern
    raw_mitre_ids = re.findall(mitre_id_pattern, text)

    # Just use valid MITRE IDs that appears in the framework
    valid_mitre_ids = mitre["ID"].astype(str).str.strip()
    mitre_ids = set(raw_mitre_ids) & set(valid_mitre_ids)

    return list(mitre_ids) # returns a valid MITRE ID set


# Returns a list of valid MITRE IDs base on a potential list of MITRE concept names
def replaceNametoID(mitre_df: pd.DataFrame, llm_response: str) -> List[str]:
    id_list = []

    for index, row in mitre_df[::-1].iterrows():  #  if sub-technique is mentioned, it will also add the "above level" technique, [::-1] iterate from bottom to top
        if row['name'] in llm_response:
            id_list.append(row['ID'])

    return id_list

# Returns a list of all occurring valid MITRE IDs and Names
def getConceptInfos(mitre_df: pd.DataFrame, llm_response: str) -> List[tuple[str, str, str]]:
    id_list = []

    for index, row in mitre_df[::-1].iterrows():
        if row['name'] in llm_response or row['ID'] in llm_response:
            id_list.append((row['ID'], row['name'], row['description']))

    return id_list



# Calculate F1, Precision, Recall and the Jaccard Index as accuracy
def calculate_metrics(prediction_list, true_label_list):
    if len(prediction_list) == 0 and len(true_label_list) == 0:  # if empty labels are correct
        return 1., 1., 1., 1.

    # unique all ids
    unique_prediction_list = list(set(prediction_list))
    unique_true_label_list = list(set(true_label_list))

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

    return float(f1), float(precision), float(recall), float(accuracy)
