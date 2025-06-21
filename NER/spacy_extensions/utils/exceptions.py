import json
from pathlib import Path
from spacy.language import Language
from spacy.pipeline.lemmatizer import Lemmatizer
from typing import Dict, List, Literal, Union

################################################################################
#                               Type definitions                               #
################################################################################

# Supported Lemmatizer tables
TypeLemmaTables = Literal["lemma_rules", "lemma_exc", "lemma_index"]

# Supported parts-of-speech, see https://universaldependencies.org/u/pos/
TypePOS = Literal[
    "adj", "adp", "adv", "aux", "cconj", "det", "intj", "noun", "num",
    "part", "pron", "propn",  "punct", "sconj", "sym", "verb", "x",
]

################################################################################
#                            Add Tagger exceptions                             #
################################################################################

def add_pos_exceptions(
        nlp: Language,
        exceptions: Dict[str, TypePOS],
    ) -> Language:
    """Set exceptions for given tokens that should have a specific POS tag.
    
        Parameters
        ----------
        nlp : Language
            NLP pipeline for which to add the POS exceptions.
            
        exceptions: Dict[str, TypePOS],
            Exceptions to add.
        """
    # Get attribute ruler component
    ruler = nlp.get_pipe("attribute_ruler")

    # Add all exceptions
    for token, pos in exceptions.items():
        ruler.add(
            [[{"LOWER": e.lower()} for e in token.split('_')]],
            {"POS": pos.upper()},
        )

    # Return nlp
    return nlp

################################################################################
#                          Add Lemmatizer exceptions                           #
################################################################################

def add_lemmatizer_exceptions(
        nlp: Language,
        rules: Dict[
                TypePOS,
                Dict[str, List[str]]
        ],
    ) -> Language:
    """Add lemmatizer exceptions to a given language using attribute rules.
        Performs matching based on lowercase version of token.
    
        Parameters
        ----------
        nlp : Language
            Pipeline for which to add Lemmatizer exceptions.

        rules : Dict[TypePos, Dict[str, List[str]]]
            Exception rules to add. Same format as Lemmatizer lemma_exc Table.

        Returns
        -------
        nlp : Language
            Pipeline where Lemmatizer exceptions have been added.
        """
    # Get attribute ruler component
    ruler = nlp.get_pipe("attribute_ruler")

    # Loop over all POS tags
    for pos, exceptions in rules.items():
        # Loop over all exceptions
        for exception, target in exceptions.items():
            # Assert we only have a single target
            assert isinstance(target, list) and len(target) == 1
            target = target[0]

            # Add exception
            ruler.add(
                [[{"LOWER": e.lower()} for e in exception.split('_')]],
                {"LEMMA": target},
            )

    # Return nlp
    return nlp
