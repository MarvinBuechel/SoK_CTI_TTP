# Imports
from collections            import Counter
from spacy.tokens           import Doc
from spacy_extensions.utils import SpacyCallable
from typing                 import Iterable
import spacy
from spacy_extensions.vocab import token2str

@spacy.Language.factory('vocab_counter')
class VocabCounter(SpacyCallable):

    def __init__(self, nlp, name, special: bool = False,):
        """SpaCy pipeline extension for counting entries in vocab.
            For each document it will add a counter to the number times each
            Lexeme in vocab occurs.

            Note
            ----
            In the current implementation we apply the ``vocab.token2str()``
            method to each word, if they are equivalent, then
            they are counted the same way.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which VocabCounter is used.

            name : string
                Name of pipeline.
            
            Configuration
            -------------
            special : bool, default=False
                If True, map IOCs and ATTACK concepts to special tokens.
            """
        # Set document
        self.nlp  = nlp
        self.name = name

        # Set extension
        Doc.set_extension(
            name   = 'vocab_count',
            getter = lambda doc: Counter(token2str(token, special=special) for token in doc),
        )

################################################################################
#                                 Total count                                  #
################################################################################

def total_count(self, documents: Iterable[Doc]) -> Counter:
    """Return total count of all analyzed documents.

        Parameters
        ----------
        documents : Iterable[Doc]
            Documents for which to get the total vocabulary count.
            All documents must be parsed with the VocabCounter.

        Returns
        -------
        counter : Counter
            Counter containing the total count of all analyzed documents.
        """
    # Return result
    return sum(map(document._.vocab_count for document in documents), Counter())
