# Imports
import argparse
import json
import pandas as pd
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from spacy_extensions.__main__ import data2report
from typing import List

# Set regex
REGEX_ATTACK = re.compile(
    r"(TA|T|DS|S|G|M)\d{4}(\.\d{3})?|ATTACK",
    re.IGNORECASE,
)

################################################################################
#                                 Evaluations                                  #
################################################################################

def eval_files(
        filenames: List[str],
        y_pred: List[dict],
        y_true: List[dict],
    ) -> None:
    """Perform and print a per-file evaluation.
    
        Parameters
        ----------
        filenames : List[str]
            List of filenames for each cti report.
            
        y_pred : List[dict]
            Predicted reports loaded from json file.
            
        y_true : List[dict]
            Manually labelled reports loaded from json file.
        """
    # Initialise metrics
    TPs, FPs, FNs = list(), list(), list()

    # Iterate over each filename, prediction, true values
    for filename, y_pred_, y_true_ in zip(filenames, y_pred, y_true):

        # Get ground-truth entity targets
        if 'ents' in y_true_:
            targets = y_true_['ents']
        else:
            targets = filter(lambda x: x, y_true_['analyzed'])
        targets = filter(lambda x: REGEX_ATTACK.match(x['label']), targets)
        targets = filter(lambda x: not x['label'].endswith('0000'), targets)
        targets = {(t['start'], t['end']): t['label'] for t in targets}

        # Check whether prediction contains ner_matches
        if '_' not in y_pred_ or 'ner_matches' not in y_pred_.get('_', {}):
            raise ValueError(
                f"The predicted doc '{filename}' does not contain custom found "
                "matches. Please check whether you input the correct documents "
                "or process the document with the specialized pipeline."
            )

        # Get predicted entities
        predictions = defaultdict(list)
        for prediction in y_pred_['_']['ner_matches']:
            if REGEX_ATTACK.match(prediction['label']):
                key = (prediction['start_char'], prediction['end_char'])
                predictions[key].append(prediction['label'])
        # for prediction in y_pred_['ents']:
        #     if REGEX_ATTACK.match(prediction['label']):
        #         key = (prediction['start'], prediction['end'])
        #         predictions[key].append(prediction['label'])
        for prediction in y_pred_['ents']:
            if prediction['label'] == 'ATTACK':
                key = (prediction['start'], prediction['end'])
                predictions[key].append(prediction['label'])

        # Compute metrics
        TP, FP = 0, 0
        for (start, end), pred_labels in predictions.items():
            # Get corresponding label
            label = targets.get((start, end))
            if label in pred_labels or 'ATTACK' in pred_labels:
                TP += 1
            else:
                FP += len(pred_labels)
        # Compute false negatives
        FN = len(set(targets.keys()) - set(predictions.keys()))

        # Add metrics to total
        TPs.append(TP)
        FPs.append(FP)
        FNs.append(FN)

    # Print report
    print(data2report({
        'name': list(map(lambda x: f"{x[:14]:14}...", filenames)),
        'TP': TPs,
        'FP': FPs,
        'FN': FNs,
    }))


def eval_techniques(
        filenames: List[str],
        y_pred: List[dict],
        y_true: List[dict],
    ) -> None:
    # Initialise metrics
    labels = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0})

    # Iterate over each filename, prediction, true values
    for filename, y_pred_, y_true_ in zip(filenames, y_pred, y_true):

        # Get ground-truth entity targets
        if 'ents' in y_true_:
            targets = y_true_['ents']
        else:
            targets = filter(lambda x: x, y_true_['analyzed'])
        targets = filter(lambda x: REGEX_ATTACK.match(x['label']), targets)
        targets = filter(lambda x: not x['label'].endswith('0000'), targets)
        targets = {(t['start'], t['end']): t['label'] for t in targets}

        # Check whether prediction contains ner_matches
        if '_' not in y_pred_ or 'ner_matches' not in y_pred_.get('_', {}):
            raise ValueError(
                f"The predicted doc '{filename}' does not contain custom found "
                "matches. Please check whether you input the correct documents "
                "or process the document with the specialized pipeline."
            )

        # Get predicted entities
        predictions = defaultdict(list)
        for prediction in y_pred_['_']['ner_matches']:
            if REGEX_ATTACK.match(prediction['label']):
                key = (prediction['start_char'], prediction['end_char'])
                predictions[key].append(prediction['label'])
        # for prediction in y_pred_['ents']:
        #     if REGEX_ATTACK.match(prediction['label']):
        #         key = (prediction['start'], prediction['end'])
        #         predictions[key].append(prediction['label'])
        for prediction in y_pred_['ents']:
            if prediction['label'] == 'ATTACK':
                key = (prediction['start'], prediction['end'])
                predictions[key].append(prediction['label'])

        # Compute metrics
        for (start, end), pred_labels in predictions.items():
            # Get corresponding label
            label = targets.get((start, end))
            if label in pred_labels:
                labels[label]['TP'] += 1
            elif 'ATTACK' in pred_labels:
                labels['ATTACK']['TP'] += 1
            else:
                for pred_label in pred_labels:
                    labels[pred_label]['FP'] += 1

        # Compute false negatives
        for (start, end), label in targets.items():
            if (start, end) not in predictions:
                labels[label]['FN'] += 1

    # Create dataframe
    index, data = zip(*sorted(labels.items()))
    df = pd.DataFrame(data=data, index=index)
    # Add total
    total = pd.DataFrame(
        index=['Total'],
        data={
            'TP': df['TP'].sum(),
            'FP': df['FP'].sum(),
            'FN': df['FN'].sum(),
    })
    df = pd.concat([df, total])

    # Compute metrics
    df['precision'] = df['TP'] / (df['TP'] + df['FP'])
    df['recall'] = df['TP'] / (df['TP'] + df['FN'])
    df['f1-score'] = 2*df['TP'] / (2*df['TP'] + df['FP'] + df['FN'])
    df['support'] = df['TP'] + df['FP'] + df['FN']
    df[pd.isna(df)] = 0

    # Print result
    print(df.to_string())


################################################################################
#                                     Main                                     #
################################################################################


def parse_args() -> argparse.Namespace:
    """Parse arguments for given program."""
    # Create parser
    parser = argparse.ArgumentParser(
        prog = "evaluate.py",
        description = "Evaluate processed with ground truth",
    )

    # Input files
    parser.add_argument('prediction',help='path to prediction directory')
    parser.add_argument('true', help='path to directory with true labels')
    
    # Parse arguments and return
    return parser.parse_args()


def main() -> None:
    """Main method executed when program is run."""
    # Parse arguments
    args = parse_args()
    
    # Read data
    y_pred = glob(str(Path(args.prediction) / '*.json'))
    y_true = glob(str(Path(args.true) / '*.json'))
    # Cast to files
    y_pred = {Path(x).stem: x for x in y_pred}
    y_true = {Path(x).stem: x for x in y_true}

    # # TODO remove
    # y_pred = {k: v for k, v in y_pred.items() if k in y_true}

    # Ensure we have complete overlap
    if not set(y_pred.keys()).issubset(set(y_true.keys())):
        keys = set(y_pred.keys()) - set(y_true.keys())
        keys = map(lambda x: f"'{x}'", keys)
        keys = ', '.join(sorted(keys))
        raise ValueError(f"Could not find predicted data for: {keys}")
    
    # Match predictions and true values
    filenames = list()
    doc_pred = list()
    doc_true = list()

    for filename, y_pred_, in y_pred.items():
        # Get corresponding y_true_
        y_true_ = y_true[filename]
        # Load data
        with open(y_pred_) as infile:
            doc_pred.append(json.load(infile))
        with open(y_true_) as infile:
            doc_true.append(json.load(infile))
        filenames.append(filename)

    # Perform evaluations
    print("Per file evaluation")
    eval_files(filenames, doc_pred, doc_true)
    print()
    print("Per ATT&CK evaluation")
    eval_techniques(filenames, doc_pred, doc_true)


if __name__ == "__main__":
    main()
