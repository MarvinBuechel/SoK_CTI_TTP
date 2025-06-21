from spacy.tokens import Span
from spacy_extensions.matcher.sentence import SentenceMatcher
from spacy_extensions.utils.tree import root2tree, subtree_view
from spacy_extensions.utils.itertools import iterable_combinations
from typing import List, Tuple, Set


class SubphraseMatcher(SentenceMatcher):
    """Subphrase matcher to match on directly dependent subphrases."""

    def _create_matches_(
            self,
            pattern: List[Set[int]],
            label: int,
            span: Span,
        ) -> List[Tuple[int, int, int]]:
        # If pattern has length 1, it is always a match
        if len(pattern) == 1:
            return [(label, index, index+1) for index in pattern[0]]

        # Initialise result
        result = list()
        # Create tree structure from given span
        tree = root2tree(span.root)

        # Set excluded pos tags
        excluded_pos = ['ADP','AUX','CCONJ','DET','PART','PRON','PUNCT','SCONJ']

        # Loop over all pattern combinations
        for combination in iterable_combinations(pattern):
            # Get tokens from combination
            tokens = [span.doc[index] for index in combination]

            # Compute subtree
            subtree = subtree_view(tree, nodes=tokens)
            # Count number of nodes in subtree that are not excluded
            weight = len([
                node for node in subtree.nodes()
                if node.pos_ not in excluded_pos or node in tokens
            ])

            # Combination is valid when all tokens are directly related
            if weight == len(tokens):
                indices = list(map(lambda x: x.i, tokens))
                result.append((label, min(indices), max(indices)+1))

        # Return result
        return result
