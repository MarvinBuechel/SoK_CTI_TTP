import json
import pandas as pd
import random

# Script for creating the TRAM2 split

TESTSPLIT_GOAL = 0.2  # Target proportion for test split

# Load data
df = pd.read_json("../cti_datasets/tram/multi_label.json")

# Create CTI reports
cti_reports = []
for index, row in df.iterrows():
    # Check if the document already exists
    existing_report = next((doc for doc in cti_reports if doc["doc_title"] == row['doc_title']), None)
    if existing_report is None:
        # Add new document
        cti_reports.append({
            "doc_title": row['doc_title'],  # Add title
            "cti_report": [row['sentence']],  # List of sentences with current sentence
            "label": [row['labels']] if row['labels'] else [[]]  # Add labels, empty list if no labels
        })
    else:
        # If the document already exists, add the sentence and labels
        existing_report["cti_report"].append(row['sentence'])
        if row['labels']:
            existing_report["label"].append(row['labels'])
        else:
            existing_report["label"].append([])  # Empty list if no labels

# Randomness
random.shuffle(cti_reports)  # Shuffle CTI reports randomly

# Calculate total number of labels
total_labels = sum(len(doc["label"]) for doc in cti_reports)

# Prepare test split
test_reports = []
test_label_count = 0
# Iteratively add reports to test split until 20% of the labels are reached
for doc in cti_reports:
    doc_label_count = len(doc["label"])
    if test_label_count + doc_label_count <= TESTSPLIT_GOAL * total_labels:
        test_reports.append(doc)
        test_label_count += doc_label_count

# Calculate training split
train_reports = [doc for doc in cti_reports if doc not in test_reports]

# Save the results
with open("../cti_datasets/tram/official_sok_split/train_split.json", "w") as train_file:
    json.dump(train_reports, train_file, indent=4)
with open("../cti_datasets/tram/official_sok_split/test_split.json", "w") as test_file:
    json.dump(test_reports, test_file, indent=4)

# Output the results
print("Training split saved in 'train_split.json'")
print("Test split saved in 'test_split.json'")
print("Proportion of labels in test split:", test_label_count / total_labels)
print("Training split size:", len(train_reports))
print("Test split size:", len(test_reports))
print("Proportion of test labels:", len(test_reports) / len(train_reports))