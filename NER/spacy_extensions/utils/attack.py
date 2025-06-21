"""Helper functions for extracting ATT&CK framework and concepts from an NLP
    pipeline."""

# Imports
from py_attack import ATTACK
from spacy.language import Language
from typing import Dict, List

def get_attack(nlp: Language) -> ATTACK:
    """Extract MITRE ATT&CK framework from given NLP pipeline.
    
        Note
        ----
        Expects nlp to contain the matcher_mitre_attack pipe.

        Parameters
        ----------
        nlp : Language
            Pipeline from which to extract ATTACK.
            
        Returns
        -------
        attack : ATTACK
            ATTACK framework used in pipeline.
        """
    # Return ATTACK pipe
    return nlp.get_pipe('matcher_mitre_attack').attack


def get_names_from_ATTACK(attack: ATTACK) -> Dict[str, List[str]]:
    """Return the concept names of MITRE ATT&CK concepts framework.
    
        Parameters
        ----------
        attack : ATTACK
            ATTACK framework from which to extract concept names.
            
        Returns
        -------
        result : Dict[str, List[str]]
            Dictionary of MITRE ATT&CK ID -> List of names.
        """
    # Filter relevant MITRE ATT&CK concepts
    concepts = filter(
        lambda concept: (concept.get('identifier') or '\0')[0] in {
            'T', # Tactics and techniques
            'S', # Software
            'G', # Groups
            'M', # Mitigations
            'D', # Data sources
        },
        attack.concepts
    )

    # Create dictionary from concepts
    result = dict()

    for concept in concepts:
        # Get identifier
        identifier = concept.get('identifier')

        # Add identifier
        if identifier not in result:
            result[identifier] = list()

        # Add names
        names = (
            [concept.get('name')] +
            concept.get('aliases', list()) +
            concept.get('x_mitre_aliases', list())
        )

        # Assert None is not found
        assert None not in names, f"{identifier} contains None"

        # Add names
        result[identifier].extend(names)

    # Return result
    return result


def get_attack_concept_names(nlp: Language) -> Dict[str, List[str]]:
    """Return the concept names of MITRE ATT&CK concepts in pipeline.
    
        Parameters
        ----------
        nlp : Language
            Pipeline from which to extract ATTACK concept names.
            
        Returns
        -------
        result : Dict[str, List[str]]
            Dictionary of MITRE ATT&CK ID -> List of names.
        """
    # Get ATTACK framework
    attack = get_attack(nlp)

    # Return names
    return get_names_from_ATTACK(attack)

    