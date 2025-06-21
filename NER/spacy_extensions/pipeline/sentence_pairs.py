# Imports
from typing import Iterator, Tuple, Union
from spacy                  import Language
from spacy.tokens           import Doc, Span
from spacy_extensions.utils import SpacyCallable

@Language.factory('sentence_pairs')
class SentencePairs(SpacyCallable):

    def __init__(self, nlp, name: str, span: bool = False):
        """SentencePairs provides a Doc extension for iterating over sentence
            pairs.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which SentencePairs is used.

            name : string
                Name of pipeline.

            Configuration
            -------------
            span : bool, default=False
                If True, return sentence pair as a single span. Otherwise,
                return sentence pair as a tuple of spans.
            """
        # Initialize pipeline
        self.nlp  = nlp
        self.name = name

        # Set configuration
        self.span = span

        # Register doc extensions
        Doc.set_extension(
            name   = "sentence_pairs",
            getter = self.sentence_pairs,
        )

        Doc.set_extension(
            name   = "sentence_pairs_skip",
            getter = self.sentence_pairs_skip,
        )


    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def sentence_pairs(self, doc: Doc) -> Iterator[Union[Span, Tuple[Span, Span]]]:
        """Provide an iterator over sentence pairs in document."""
        # Set previous sentence reference
        previous = None

        # Loop over sentences in document
        for sentence in doc.sents:
            # Yield (previous, current) sentence pair
            if previous is not None:
                # Return sentence pair as Span if required
                if self.span:
                    yield doc[previous.start : sentence.end]
                # Otherwise (default), return as Tuple
                else:
                    yield (previous, sentence)
            
            # Set previous sentence to current 
            previous = sentence


    def sentence_pairs_skip(self, doc: Doc) -> Iterator[Union[Span, Tuple[Span, Span]]]:
        """Provide an iterator over sentence pairs in document, skipping the
            second occurence. E.g., in the document "We have a dog. The dog eats
            food. Then he sleeps. We don't have a cat." will return only two
            skipped sentence pairs instead of three:
            - ("We have a dog.", "The dog eats food.")
            - ("Then he sleeps.", "We don't have a cat.")
            """
        for index, sentence_pair in enumerate(self.sentence_pairs(doc)):
            if index & 1 == 0:
                yield sentence_pair