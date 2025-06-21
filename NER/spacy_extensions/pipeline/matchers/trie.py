# Imports
from __future__ import annotations
from typing     import Iterable, List, Literal, Optional, Tuple
import json
import spacy

# Local imports
from spacy_extensions.utils import SpacyCallable, SpacySerializable, Trie

################################################################################
#                                 MatcherTrie                                  #
################################################################################

@spacy.Language.factory('matcher_trie')
class MatcherTrie(SpacyCallable):

    def __init__(
            self,
            nlp,
            name,
            sequences: List[Tuple[str, str]] = list(),
            ignore_case   : bool = False,
            alignment_mode: Literal['strict', 'contract', 'expand'] = 'strict',
        ):
        """SpaCy pipeline extension for matching NER based on a Trie.
            MatcherTrie is used as an alternative to the MatcherRegex. Matching
            with a Trie uses a datastructure that performs a quicker lookup.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which MatcherTrie is used.

            name : string
                Name of pipeline.

            Config Parameters
            -----------------
            sequences : List[Tuple[str, str]], default=list()
                Sequences with which to fit MatcherTrie.

            ignore_case : bool, default=False
                If True, the case of strings is ignored by the MatcherTrie.

            alignment_mode : Literal['strict', 'contract', 'expand'], default="strict"
                See alignment_mode of https://spacy.io/api/doc#char_span
            """
        # Initialise super
        super().__init__()

        # Set document
        self.nlp  = nlp
        self.name = name

        # Add configuration
        self.alignment_mode = alignment_mode

        # Create and fittrie
        self.trie = Trie(ignore_case=ignore_case)
        self.trie.fit(*zip(*sequences))

    ########################################################################
    #                                 Call                                 #
    ########################################################################

    def __call__(self, document):
        """Identify entities in text based on the fitted underlying Trie.

        Parameters
        ----------
        document : spacy.tokens.doc.Doc
            Document in which to identify entities.

        Returns
        -------
        document : spacy.tokens.doc.Doc
            Document where entities are identified using Trie-based NER.
        """
        # Collect all spans
        spans = list()

        # Loop over all matches
        for start, end, label in self.trie.predict(document.text):
            # Add corresponding span in document
            span = document.char_span(
                start,
                end,
                label          = ' | '.join(label),
                alignment_mode = self.alignment_mode,
            )

            # Collect span
            spans.append(span)

        # Set document entities
        document.set_ents(
            spacy.util.filter_spans(
                tuple(filter(None, spans)) +
                document.ents
            )
        )

        # Return document
        return document
