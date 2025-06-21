# Imports
import spacy
from spacy.tokens import Span
from typing import Any, Dict, List, Optional
from spacy_extensions.pipeline.matchers.sentence import SentenceMatcher
from spacy_extensions.pipeline.matchers.matcher import AttributeMatcher

################################################################################
#                                     Pipe                                     #
################################################################################

@spacy.Language.factory('matcher_related', requires=[
    'token._.token_base',
    'token._.related_base',
])
class MatcherRelated(SentenceMatcher):
    """Entity matcher for related items that have been trained on."""

    def __init__(
            self,
            nlp,
            name,
            exclude_pos: List[str]=[
                'ADP', 'AUX', 'CCONJ', 'DET', 'PART', 'PRON', 'SCONJ'
            ],
        ):
        # Initialise super
        super().__init__(nlp, name)
        # Overwrite matcher
        self.matcher = AttributeMatcher(
            vocab     = self.nlp.vocab,
            attribute = 'token_base',
            custom    = True,
            relative  = True,
        )
        # Configure excluded POS
        self.excluded_pos = set(exclude_pos)

    ########################################################################
    #                              Fit method                              #
    ########################################################################

    def add_sentences(
            self,
            match_id: str,
            sentences: List[List[Dict[str, Any]]],
            greedy: Optional[str] = None,
        ) -> None:
        """Fit the MatcherRelated with samples."""
        return self.add(
            match_id = match_id,
            patterns = [self.pattern(sentence) for sentence in sentences],
            greedy   = greedy,
        )


    def pattern(self, sentence: Span) -> List[dict]:
        """Create a matcher pattern for a given sentence span.
        
            Parameters
            ----------
            sentence : Span
                Sentence for which to create a matching pattern.
                
            Returns
            -------
            List[dict]
                Pattern on which to match.
            """
        # Disable own pipe
        self.nlp.disable_pipe(self.name)

        # Create patterns
        result = list()

        for token in self.nlp(sentence):
            if token.pos_ not in self.excluded_pos:
                # Get token base
                token_base = list(token._.related_base)
                
                # Add token base, case single token_base
                if len(token_base) == 1:
                    result.append({
                        "_": {"token_base": token_base[0]},
                    })
                
                # Add token base, case multiple token_bases
                else:
                    result.append({
                        "_": {"token_base": {"IN": list(token._.related_base)}},
                    })

        # Re-enable own pipe
        self.nlp.enable_pipe(self.name)

        # Return result
        return result