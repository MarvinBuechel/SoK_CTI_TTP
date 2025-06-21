from typing import Iterable
from spacy.tokens           import Doc, Span
from spacy_extensions.utils import SpacyCallable
import spacy

@spacy.Language.factory('paragraph_segmenter')
class ParagraphSegmenter(SpacyCallable):
    """ParagraphSegmenter segments documents into paragraphs.
        The API is similar to the default SpaCy
        `sentencizer <https://spacy.io/api/sentencizer>`_.
        
        Assigned Attributes
        -------------------
        ``Doc._.paragraphs``: Iterable[Span]
            Iterator over paragraphs within ``Doc``.
        """
    
    def __init__(self, nlp, name: str):
        """Segment paragraphs within documents.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which ParagraphDetector is used.

            name : string
                Name of pipeline.
            """
        # Initialize pipeline
        self.nlp  = nlp
        self.name = name

        Doc.set_extension(
            name   = "paragraphs",
            getter = self.paragraphs,
        )
    

    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def paragraphs(self, doc: Doc) -> Iterable[Span]:
        """Iterate over paragraphs in doc."""
        # Initialize paragraph start
        start = None

        # Loop over sentences in document
        for sentence in doc.sents:
            # Set start
            if start is None:
                start = sentence.start

            # Stop when sentence end is reached
            if sentence.end >= len(doc):
                yield doc[start:sentence.end-1]
                return

            # Yield paragraph
            if doc[sentence.end].text.count('\n') >= 2:
                yield doc[start:sentence.end]
                start = sentence.end+1