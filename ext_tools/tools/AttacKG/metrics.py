def calculate_metrics(pred_list, true_list, metric: str):
    # unique all ids

    unique_prediction_list = list(set(pred_list))
    unique_true_label_list = list(set(true_list))

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
