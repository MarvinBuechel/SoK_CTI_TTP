# Imports
from spacy        import Language
from spacy.tokens import Doc, Token
from spacy_extensions.utils import SpacyCallable
from typing import List
import numpy as np

@Language.factory('contextualizer_sentence')
class ContextualizerSentence(SpacyCallable):
    """ContextualizerSentence for creating token context.
        Many word embedding methods require a context surrounding a word to
        learn an embedding representation. The ContextualizerSentence provides
        the extensions for defining such contexts.

        Extensions
        ----------
        context_sentence : Token
            Returns the sentence context of a token. I.e., a numpy array
            containing all tokens within the sentence.
            See :py:meth:`get_context`.

        context_sentences : Doc
            Returns the sentence contexts of all sentences in doc.
            See :py:meth:`get_context_doc`.

        sent_index : Token
            Returns the index of a token within its sentence.
            See :py:meth:`get_sent_index`.
        """

    def __init__(self, nlp, name: str):
        """ContextualizerSentence for creating token context.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which Contextualizer class is used.

            name : string
                Name of pipeline.
            """
        # Initialize pipeline
        self.nlp  = nlp
        self.name = name

        # Set extensions
        Token.set_extension(
            name   = 'context_sentence',
            getter = self.get_context,
        )
        Doc.set_extension(
            name   = 'context_sentences',
            getter = self.get_context_doc,
        )
        Token.set_extension(
            name   = 'sent_index',
            getter = self.get_sent_index,
        )

    ########################################################################
    #                           Compute context                            #
    ########################################################################

    def get_context(self, token: Token) -> np.ndarray:
        """Retrieve the token context as a numpy array.

            Parameters
            ----------
            token : spacy.tokens.Token
                Token for which to retrieve the context.

            Returns
            -------
            context : np.array of shape=(_context_before + _context_after)
                Context for token in the document. Concatenation of
                context_before and context_after tokens. To get the
                context_before and context_after separately, simply call
                context[:, :context_before] or context[:, -context_after:]
                respectively.
            """
        # Return entire sentence as context
        return np.asarray(token.sent)

    def get_context_doc(self, doc: Doc) -> List[np.ndarray]:
        """Retrieve the sentence contexts as a numpy array.
        
            Parameters
            ----------
            doc : spacy.tokens.Doc
                Doc for which to retrieve sentence contexts.
            
            Returns
            -------
            contexts : List[np.ndarray]
                Context for each sentence in doc.
            """
        # Return entire doc as list of numpy arrays
        return [np.asarray(sentence) for sentence in doc.sents]

    ########################################################################
    #                            Sentence index                            #
    ########################################################################

    def get_sent_index(self, token: Token) -> int:
        """Retrieve the token index in a sentence.

            Parameters
            ----------
            token : spacy.tokens.Token
                Token for which to retrieve the sentence index.

            Returns
            -------
            index : int
                Index of token in sentence.
            """
        # Get context from token
        for index, tok in enumerate(token._.context):
            if isinstance(tok, Token) and tok == token:
                return index
