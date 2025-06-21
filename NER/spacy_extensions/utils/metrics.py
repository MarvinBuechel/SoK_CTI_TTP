"""Metrics to compare SpaCy Docs."""

# Imports
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Literal, Optional, Union
from sklearn.metrics import classification_report
from spacy.tokens import Doc
import numpy as np
import pandas as pd
import warnings

# Create metrics type
Metrics = Literal[
    'precision',
    'recall',
    'f1-score',
    'accuracy',
    'support',
]

################################################################################
#                                   Reports                                    #
################################################################################

def classification_report_docs(
        y_true: Iterable[Doc],
        y_pred: Iterable[Doc],
        underscore: Optional[List[str]] = None,
        digits: int = 4,
        output_dict: bool = False,
    ) -> Union[Dict[Literal['overall', 'reports'], Any], str]:
    """Build a text report showing the main classification metrics for 
        comparing multiple Doc reports with their ground truth version.

        Parameters
        ----------
        y_true: Iterable[Doc]
            SpaCy Docs containing ground truth labels.

        y_pred: Iterable[Doc]
            SpaCy Docs with predicted labels.

        underscore : Optional[List[str]], default=None
            Token underscore attributes to include in report.

        digits : int, default=4
            Number of digits to print in resulting report

        output_dict : bool, default=False
            If true, output dictionary.

        Returns
        -------
        result : Union[Dict[Literal['overall', 'reports'], Any], str]
            Resulting classification report(s) in dict or text format.
        """
    # Initialise individual reports
    reports = list()

    # Create individual doc reports
    for y_true_, y_pred_ in zip(y_true, y_pred):
        reports.append(classification_report_doc(y_true_, y_pred_))

    # Merge all values to create overall report
    overall = defaultdict(lambda: {
        'true': list(),
        'pred': list(),
    })

    # Merge reports
    for report in reports:
        for attribute, values in report.items():
            overall[attribute]['true'].extend(values['true'])
            overall[attribute]['pred'].extend(values['pred'])

    # Create report
    for attribute, values in overall.items():
        overall[attribute]['report'] = classification_report(
            y_true = values['true'],
            y_pred = values['pred'],
            output_dict = True,
            zero_division = 0,
        )

    # Create result
    result = {
        'overall': overall,
        'reports': reports,
    }

    # Output dictionary or string
    if output_dict:
        return result
    else:
        return report2string(overall, digits=digits)


def classification_report_doc(
        y_true: Doc,
        y_pred: Doc,
        underscore: Optional[List[str]] = None,
    ) -> Dict[str, Dict[Literal['true', 'pred', 'report'], Any]]:
    """Create a classification report for a single Doc comparison.
    
        Parameters
        ----------
        y_true : Doc
            Document containing ground truth values for tokens and entities.

        y_pred : Doc
            Document containing predicted values for tokens and entities.

        underscore : Optional[List[str]], default=None
            Token underscore attributes to include in report.

        Returns
        -------
        result : Dict[str, Dict[Literal['true', 'pred', 'report'], Any]]
            Classification report dictionary.
        """
    # Retrieve attributes
    result = compare_token_attributes(y_true, y_pred, underscore)
    result['ent'] = compare_entities(y_true, y_pred)

    # Create report
    for attribute, values in result.items():
        result[attribute]['report'] = classification_report(
            y_true = values['true'],
            y_pred = values['pred'],
            output_dict = True,
            zero_division = 0,
        )

    # Return result
    return result


################################################################################
#                              Compare functions                               #
################################################################################

def compare_token_attributes(
        y_true: Doc,
        y_pred: Doc,
        underscore: Optional[List[str]] = None,
    ) -> Dict[str, Dict[Literal['true', 'pred'], np.ndarray]]:
    """Compare the attributes of tokens in each document.
    
        Parameters
        ----------
        y_true : Doc
            Document containing ground truth values for tokens.

        y_pred : Doc
            Document containing predicted values for tokens.

        underscore : Optional[List[str]], default=None
            Token underscore attributes to include in comparison.

        Returns
        -------
        attributes : Dict[str, Dict[Literal['true', 'pred'], np.ndarray]]
            Dictionary containing entry for each token attribute present in the
            document. Each entry contains a dictionary of:
            - 'true' -> np.array
            - 'pred' -> np.array
            Where the 'true' entry contains all ground truth values and the 
            'pred' entry contains all predicted values.
        """
    # Check whether docs are comparable
    if not check_comparable(y_true, y_pred):
        raise ValueError("Docs are not comparable")

    # Initialise all attributes
    attributes = defaultdict(lambda: {
        'true': list(),
        'pred': list(),
    })

    # Loop over all tokens
    for token_true, token_pred in zip(y_true, y_pred):

        for pred, token in [("true", token_true), ("pred", token_pred)]:
            # Add default attributes
            attributes["tag"  ][pred].append(token.tag_)
            attributes["pos"  ][pred].append(token.pos_)
            attributes["morph"][pred].append(token.morph.to_json())
            attributes["lemma"][pred].append(token.lemma_)
            attributes["dep"  ][pred].append(token.dep_)
            attributes["head" ][pred].append(token.head.i)

            # Add underscore attributes
            for attr in underscore or []:
                attributes[attr][pred].append(getattr(token._, attr))

    # Transform attributes into numpy arrays
    for attr in attributes:
        attributes[attr]['true'] = np.asarray(attributes[attr]['true'])
        attributes[attr]['pred'] = np.asarray(attributes[attr]['pred'])

        # Check dtype
        if str(attributes[attr]['true'].dtype).startswith('<U'):
            attributes[attr]['true'] = attributes[attr]['true'].astype(object)
        if str(attributes[attr]['pred'].dtype).startswith('<U'):
            attributes[attr]['pred'] = attributes[attr]['pred'].astype(object)

        # Assert dtypes are the same
        assert attributes[attr]['true'].dtype == attributes[attr]['pred'].dtype

    # Return attributes
    return dict(attributes)


def compare_entities(
        y_true: Union[Doc, Dict[str, Any]],
        y_pred: Union[Doc, Dict[str, Any]],
    ) -> Dict[Literal['true', 'pred'], np.ndarray]:
    """Compare the entities in each document.
        
        Note
        ----
        In its current form, entity comparison is strict, meaning that entities
        are only considered equivalent if they have the exact same start and end
        position.
    
        Parameters
        ----------
        y_true : Union[Doc, Dict[str, Any]]
            Document containing ground truth values for entities.

        y_pred : Union[Doc, Dict[str, Any]]
            Document containing predicted values for entities.

        Returns
        -------
        ents : Dict[Literal['true', 'pred'], np.ndarray]
            Dictionary containing true and pred values that can be compared 
            using sklearn.metrics:
            - 'true' -> np.array
            - 'pred' -> np.array
            Where the 'true' entry contains all ground truth values and the 
            'pred' entry contains all predicted values.
        """
    # Transform Doc to JSON
    y_true = as_json(y_true)
    y_pred = as_json(y_pred)

    # Check whether docs are comparable
    if not check_comparable(y_true, y_pred):
        raise ValueError("Docs are not comparable")

    # Sort entities
    ent_true = {(ent['start'], ent['end']): ent for ent in y_true['ents']}
    ent_pred = {(ent['start'], ent['end']): ent for ent in y_pred['ents']}
    
    # Get list of merged ents
    ents_merged = list(sorted(set([
        (ent['start'], ent['end']) for ent in y_true['ents'] + y_pred['ents']
    ])))

    # Create output
    result_true = []
    result_pred = []
    for ent in ents_merged:
        result_true.append(ent_true.get(ent, {}).get('label', np.NaN))
        result_pred.append(ent_pred.get(ent, {}).get('label', np.NaN))

    # Return result
    return {
        'true': np.asarray(result_true),
        'pred': np.asarray(result_pred),
    }
        
################################################################################
#                                    Checks                                    #
################################################################################

def check_comparable(
        a: Union[Doc, Dict[str, Any]],
        b: Union[Doc, Dict[str, Any]],
    ) -> bool:
    """Assert whether two SpaCy Docs are comparable.
        Raises an error in case documents are not comparable.
    
        Parameters
        ----------
        a : Doc
            First document to compare.

        b : Doc
            Second document to compare.

        Returns
        -------
        comparable : bool
            True if Docs a and b can be compared.
        """
    # Transform Doc to JSON
    a = as_json(a)
    b = as_json(b)

    # Check whether both docs have the same text
    if a['text'] != b['text']:
        return False

    # Check whether we have the same amount of tokens
    elif len(a['tokens']) != len(b['tokens']):
        return False

    # Check whether we have the same amount of sents
    elif len(a['sents']) != len(b['sents']):
        return False

    # Check whether all tokens are the same
    for a_, b_ in zip(a['tokens'], b['tokens']):
        # Check whether each token covers the same text span
        if a_['start'] != b_['start'] or a_['end'] != b_['end']:
            return False
    
    # Check whether all sentences are the same
    for a_, b_ in zip(a['sents'], b['sents']):
        # Check whether each sent covers the same text span
        if a_['start'] != b_['start'] or a_['end'] != b_['end']:
            return False

    # Everything is comparable
    return True

################################################################################
#                             Auxiliary functions                              #
################################################################################

def as_json(doc: Union[Doc, Dict[str, Any]]) -> Dict[str, Any]:
    """Transform a given doc to JSON if it is not in JSON format.
    
        Parameters
        ----------
        doc : Union[Doc, Dict[str, Any]]
            Doc to transform to JSON. If it is already in JSON format, do
            nothing.
            
        Returns
        -------
        json : Dict[str, Any]
            JSON representation of Doc.
        """
    # If a Doc is given, return JSON version of doc
    if isinstance(doc, Doc):
        doc = doc.to_json()
    # If a non-supported type is given raise error
    elif not isinstance(doc, dict):
        raise TypeError(
            "Expected 'doc' type 'spacy.tokens.Doc' or dict, but was of type "
            f" {type(doc)}."
        )
    
    # Ensure that doc representation has the expected keys
    keys = {'text', 'ents', 'sents', 'tokens'}
    if set(doc.keys()) != keys:
        raise ValueError(
            f"Expected doc to have keys: '{keys}', but had keys "
            f"'{set(doc.keys())}'."
        )

    # Return doc
    return doc


def report2string(
        report: Dict[str, Any],
        digits: int = 2,
        minimized: bool = True,
    ) -> str:
    """Create string representation of report.
    
        Parameters
        ----------
        report : Dict[str, Any]
            Report to transform into string.

        digits : int, default=2
            Number of digits to print by default.

        minimized : bool, default=True
            If True, only return a minimized version of the report.
        
        Returns
        -------
        result : str
            String representation of report.
        """
    # Initialise result
    result = list()

    # Loop over all attributes
    for attribute, values in report.items():
        # Get report
        for key, metrics in values['report'].items():
            if key == 'accuracy':
                metrics = {
                    '': attribute,
                    'value': key,
                    'f1-score': metrics
                }
            else:
                # Add attribute
                metrics[''] = attribute
                metrics['value'] = key

            # Add to dataframe
            result.append(metrics)
        result.append({})

    # Cast to dataframe
    result = pd.DataFrame(result)
    # Reorder dataframe
    result = result[['', 'value', 'precision', 'recall', 'f1-score', 'support']]

    # Only return weighted avg if minimized
    if minimized:
        result = result[result['value'] == 'weighted avg']
        result = result[['', 'precision', 'recall', 'f1-score', 'support']]

    # Return as string
    return result.to_string(
        index = False,
        na_rep = '',
        float_format = lambda x: f"{x:.{digits}f}",
    )
