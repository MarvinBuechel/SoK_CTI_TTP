# Imports
import json
import warnings
import spacy
from collections import defaultdict
from typing import Dict, Iterable, List, Optional
from py_attack import ATTACK
from py_attack.types import DomainTypes
from spacy.tokens import Doc, Span, Token
from spacy.matcher import DependencyMatcher
from spacy_extensions.utils import SpacyCallable, SpacySerializable
from spacy_extensions.utils.matcher import AttackMixin
from tqdm import tqdm

################################################################################
#                                     Pipe                                     #
################################################################################

@spacy.Language.factory('matcher_fuzzy')
class MatcherFuzzy(SpacyCallable, SpacySerializable, AttackMixin):
    """In contrast to the PhraseMatcher, a MatcherFuzzy is able to detect tokens
        a single phrase that for which the word order is not important, but the
        dependencies are important. E.g., if we want to detect "data
        exfiltration" then the sentence "Data was exfiltrated over the network"
        should be detected, even though the exact sequence "data exfiltration"
        is not present. The MatcherFuzzy takes care of such matching.
            
        Requires
        --------
        matcher_dict : Token
            Matcher dict for matching token, provided by spacy_extensions.
            pipeline.TokenBase.
        """
    
    def __init__(self, nlp, name):
        """Create an entity ruler for the MITRE ATT&CK concepts."""
        # Set base attributes
        self.nlp  = nlp
        self.name = name

        # Create dependency matcher
        self.matcher  = DependencyMatcher(self.nlp.vocab)
        self.patterns = list()

    ########################################################################
    #                                 Call                                 #
    ########################################################################


    def __call__(self, doc: Doc) -> Doc:
        """Find patterns"""
        # Get matching spans
        spans = self.matches(doc)

        # Set document entities
        doc.set_ents(
            spacy.util.filter_spans(
                tuple(filter(None, spans)) +
                doc.ents
            )
        )

        # Return document
        return doc

    ########################################################################
    #                              Extensions                              #
    ########################################################################
    
    def matches(self, doc: Doc) -> List[Span]:
        """Find matches of spans within the given document.
        
            Parameters
            ----------
            doc : Doc
                Document in which to find matches.
                
            Returns
            -------
            spans : List[Span]
                Spans matching the fuzzy_matcher patterns.
            """
        # Find matches
        matches = self.matcher(doc)

        # Create spans
        spans = list()
        for label, span in matches:
            spans.append(Span(
                doc = doc,
                start = min(span),
                end   = max(span)+1,
                label = label,
            ))

        # Filter overlapping spans with same label by smallest dependency tree
        spans = self.filter_spans(spans)

        # Return spans
        return spans


    def filter_spans(self, spans: List[Span]) -> List[Span]:
        """Filter overlapping spans with same label by smallest distance in
            dependency tree.
        
            Parameters
            ----------
            spans : List[Span]
                List of spans to filter.

            Returns
            -------
            result : List[Span]
                Filtered spans.
            """
        warnings.warn("Filter spans not correctly implemented yet")

        return spans

        # Dictionary of label -> spans
        span_dict = defaultdict(list)
        for span in spans:
            span_dict[span.label].append(span)

        # Initialise result
        result = dict()

        # Loop over all spans
        for label, spans_ in span_dict.items():
            # Set initial span
            result[label] = spans_[0]

            # Loop over all other spans
            for span in spans_[1:]:
                result[label] = min(result[label], span, key=lambda x: len(x))

        # Return result
        return list(result.values())

    ########################################################################
    #                                 Fit                                  #
    ########################################################################

    def add_phrases(
            self,
            X: Iterable[str],
            y: Iterable[str],
            verbose: bool = True,
        ) -> None:
        """Transform provided phrases and labels into patterns and add them to
            the fuzzy matcher.
            
            Parameters
            ----------
            X : Iterable[str] of size=(n_patterns,)
                Patterns on which to match.
                
            y : Iterable[str] of size=(n_patterns,)
                Labels on which to match.

            verbose : bool, default=False
                If true, print progress
            """
        # Disable pipe, if required
        enabled = self.name in dict(self.nlp.pipeline)
        if enabled: self.nlp.disable_pipe(self.name)

        # Add verbosity if necessary
        iterator = zip(self.nlp.pipe(X), y)
        if verbose:
            iterator = tqdm(iterator, desc='Adding fuzzy patterns')

        # Loop over patterns
        for tokens, label in iterator:

            # Set software and groups to proper nouns
            if label.startswith('S') or label.startswith('G'):
                for token in tokens:
                    token._.pos = "PROPN"

            # Create patterns
            patterns = list()
            # for operator in ['<<', '>>', '.*', ';*']:
            for operator in ['<<', '>>']:
                # Create pattern from tokens
                pattern = [{
                    "RIGHT_ID": "base",
                    "RIGHT_ATTRS": tokens[0]._.matcher_dict,
                }]

                for index, token in enumerate(tokens[1:]):
                    pattern.append({
                        "LEFT_ID": "base",
                        "REL_OP": operator,
                        "RIGHT_ID": str(index),
                        "RIGHT_ATTRS": token._.matcher_dict,
                    })

                # Add pattern
                patterns.append(pattern)

            # Add patterns
            self.matcher.add(label, patterns)
            self.patterns.append({
                'label'   : label,
                'patterns': patterns,
            })

        # Re-enable pipe
        if enabled: self.nlp.enable_pipe(self.name)

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_bytes(self, exclude: Optional[Iterable[str]] = None) -> bytes:
        """Return learned patterns as json."""
        return json.dumps(self.patterns).encode('utf-8')
        

    def from_bytes(self, data: bytes, exclude: Optional[Iterable[str]] = None):
        """Load matcher from bytes."""
        # Load patterns
        self.patterns = json.loads(data.decode('utf-8'))

        # Create new dependency matcher
        self.matcher  = DependencyMatcher(self.nlp.vocab)

        # Add patterns to matcher
        for pattern in self.patterns:
            # Check if pattern is correct
            if 'label' not in pattern:
                raise ValueError("Pattern did not contain label.")
            if 'patterns' not in pattern:
                raise ValueError("Pattern did not contain patterns.")
            
            # Add to matcher
            self.matcher.add(pattern['label'], pattern['patterns'])

        # Return self
        return self
