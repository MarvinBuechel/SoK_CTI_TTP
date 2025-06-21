from pathlib import Path
from py_attack import ATTACK, DomainTypes
from spacy.tokens import Span, Token
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


def sentence2pattern(
        sentence: Span,
        token_pattern: Callable[[Token], Dict[str, Any]],
        exclude_pos: List[str] = ['ADP','AUX','CCONJ','DET','PART','PRON', 'PUNCT', 'SCONJ'],
    ) -> List[Dict[str, Any]]:
    """Transform a parsed sentence to a detection pattern.
    
        Parameters
        ----------
        sentence : Span
            Sentence to parse.

        token_pattern : Callable[[Token], Dict[str, Any]]
            Function that creates the pattern for individual tokens.
            
        Returns
        -------
        result : List[Dict[str, Any]]
            Pattern representing given sentence.
        """
    # Initialise result
    result = list()

    # Loop over tokens in sentence
    for token in sentence:
        # Check if token should be included
        if token.pos_ not in exclude_pos:
            # Create pattern and append
            result.append(token_pattern(token))

    # Return pattern
    return result


def pattern_related(token: Token) -> Dict[str, Any]:
    """Create pattern for individual tokens.
        This implementation matches on given token base.
    
        Parameters
        ----------
        token : Token
            Token for which to create pattern.
            
        Returns
        -------
        result : Dict[str, Any]
            Token pattern generated from given token.
        """
    # Get base
    base = list(token._.related_base)

    # Case of single base
    if len(base) == 1:
        return {"_": {"token_base": base[0]}}
    else:
        return {"_": {"token_base": {"IN": base}}}


def cti2phrases(
        path: Optional[str] = None,
        url : Optional[str] = None,
        domains: List[DomainTypes] = ['enterprise'],
    ) -> Iterable[Tuple[str, str]]:
    """Load MatcherMitreAttack using an attack framework from either a given
        ``path`` or ``url``.

        Note
        ----
        This method loads phrases from a given ``path`` or ``url`` pointing
        to a MITRE ATT&CK framework. If you instead want to load phrases
        from a ATTACK object directly, use the :py:meth`from_attack` method
        instead.

        Parameters
        ----------
        path : Optional[str]
            Path from which to load ATTACK framework.
            If this is given, it will always prefer to load via path over URL.
            
        url : Optional[str]
            URL from which to load ATTACK framework.
            If this is given, it will always prefer to load via path over URL.
            
        domains : List[DomainTypes], default=['enterprise']
            ATTACK domains to include when loading.

        Returns
        -------
        self : self
            Returns self
        """
    # Load ATT&CK framework from path, if given
    if path is not None:
        # Case of non-complete path
        if not path.endswith('.json'):
            path = str(Path(path) / '{domain}-attack' / '{domain}-attack.json')
        # Load attack
        attack = ATTACK.load(
            path    = path,
            domains = domains,
        )

    # Download ATT&CK framework from URL, if given
    elif url is not None:
        attack = ATTACK.download(
            url     = path,
            domains = domains,
        )

    # Otherwise, throw error
    else:
        raise ValueError(
            "Please configure either a path or url for the MITRE ATT&CK CTI "
            "framework to use in the AttackMixin class."
        )

    # Load self from ATT&CK
    return attack2phrases(attack)


def attack2phrases(attack: ATTACK) -> Iterable[Tuple[str, str]]:
    """Extract phrases and labels describing each ATT&CK concept from the given
        ATTACK framework.
    
        Parameters
        ----------
        attack : ATTACK
            MITRE ATT&CK framework from which to extract phrases and labels.

        Yields
        ------
        X : str
            Phrase pattern (str) on which to match.
            Can be used as input for ``MatcherPhrases.add_phrases``.

        y : str
            Label corresponding to phrase pattern on which to match.
            Can be used as input for ``MatcherPhrases.add_phrases``.
        """
    # Filter relevant concepts
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

    # Loop over all concepts
    for concept in concepts:
        # Get concept label and name
        label = concept.get('identifier')
        name  = concept.get('name')
        # Get all aliases
        aliases = set(
            [name, label] +
            concept.get('aliases', list()) +
            concept.get('x_mitre_aliases', list())
        )

        # Loop over each alias as if it were a Doc
        for alias in aliases:
            # Skip special case 'at'
            if alias == 'at' and label == 'S0110': continue
            if label == 'S1053.001': continue
            if label == 'S1053.002': continue

            # Yield labels and aliases
            yield alias, label
