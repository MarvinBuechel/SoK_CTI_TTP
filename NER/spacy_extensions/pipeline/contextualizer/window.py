# Imports
from spacy        import Language
from spacy.tokens import Doc, Token
from spacy_extensions.utils import SpacyCallable
from typing       import Optional, Tuple, Union
import numpy as np

@Language.factory('contextualizer_window')
class ContextualizerWindow(SpacyCallable):
    """ContextualizerWindow for creating token context windows.
        Many word embedding methods require a context window surrounding a word
        to learn an embedding representation. The ContextualizerWindow provides
        such a context window based on a configured window size.
        
        Extensions
        ----------
        context_window : Token
            Returns the context window of a token. See :py:meth:`get_context`.

        context_windows : Doc
            Returns the context window of all tokens in doc.
            See :py:meth:`get_context_doc`.
        """

    def __init__(
            self,
            nlp,
            name: str,
            context: Union[int, Tuple[int, int]] = (5, 5),
            no_token: Optional[float] = None,
            sentence_only: bool = False,
        ):
        """ContextualizerWindow for creating token context windows.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which ContextualizerWindow is used.

            name : string
                Name of pipeline.

            Configuration
            -------------
            context : Union[Tuple[int, int], int], default=(5, 5)
                The context can be given as an (int, int) tuple describing the
                context size before the word and after the word, respectively.
                Alternatively, when setting the context with an int X, this will
                be equivalent to setting the context to (X, X).

            no_token : float, default=None
                Token to return as context in case no context is found.

            sentence_only : boolean, default=True
                If True , only take the context window within a single sentence.
                If False, also take into account the context window outside
                sentences. E.g., "The dog eats food. Then it went for a walk."
                The (2,2)-context window of 'eats' would be:
                - sentence_only = True : [The, dog, food, <no_token>]
                - sentence_only = False: [The, dog, food, Then]
            """
        # Initialize pipeline
        self.nlp  = nlp
        self.name = name

        # Initialise configuration
        self._context_before = -1
        self._context_after  = -1
        self.no_token        = no_token
        self.sentence_only   = sentence_only
        self.context         = context

        # Set extensions
        Token.set_extension(
            name   = 'context_window',
            getter = self.get_context,
        )
        Doc.set_extension(
            name   = 'context_windows',
            getter = self.get_context_doc,
        )


    ########################################################################
    #                            Configuration                             #
    ########################################################################

    @property
    def context(self) -> Tuple[int, int]:
        """The context window used for the contextualizer."""
        # Return left and right context
        return (self._context_before, self._context_after)


    @context.setter
    def context(self, value: Union[Tuple[int, int], int]) -> None:
        """Set the context window of the contextualizer.

            Parameters
            ----------
            value : Union[Tuple[int, int], int], default=(5, 5)
                The context can be given as an (int, int) tuple describing the
                context size before the word and after the word, respectively.
                Alternatively, when setting the context with an int X, this will
                be equivalent to setting the context to (X, X).
            """
        # Case of int
        if isinstance(value, int):
            self._context_before = value
            self._context_after  = value

        # Case of Tuple[int, int]
        elif (
                (isinstance(value, tuple) or isinstance(value, list)) and
                len(value) == 2 and
                isinstance(value[0], int) and
                isinstance(value[1], int)
            ):
            self._context_before = value[0]
            self._context_after  = value[1]

        # Exception
        else:
            raise ValueError(
                f"Value '{value}' should be an int or an (int, int) tuple."
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
        # Return context depending on sentence_only
        if self.sentence_only:
            return self._get_context_sentence(token)
        else:
            return self._get_context_full_document(token)


    def get_context_doc(self, document: Doc) -> np.ndarray:
        """Retrieve the token contexts for all tokens in document as np array.

            Parameters
            ----------
            document : spacy.tokens.Doc
                Document for which to retrieve the contexts.

            Returns
            -------
            context : np.array of shape=(n_tokens, _context_before + _context_after)
                Context for tokens in the document. Concatenation of
                context_before and context_after tokens. To get the
                context_before and context_after separately, simply call
                context[:, :context_before] or context[:, -context_after:]
                respectively.
            """
        # Return context depending on sentence_only
        if self.sentence_only:
            return self._get_context_doc_sentence(document)
        else:
            return self._get_context_doc_full_document(document)


    def _get_context_sentence(self, token: Token) -> np.ndarray:
        """Compute the sentence context for a token in the document.
            Used when self.sentence_only == True.

            Parameters
            ----------
            token : spacy.tokens.Token
                Token for which to generate the context.

            Returns
            -------
            context : np.array of shape=(_context_before + _context_after)
                Context for token in the document. Concatenation of
                context_before and context_after tokens. To get the
                context_before and context_after separately, simply call
                context[:, :context_before] or context[:, -context_after:]
                respectively.
            """
        # Get sentence of token as numpy array
        sentence = np.asarray(token.sent)

        # Get index of token
        index = np.where(sentence == token)[0][0]

        # Get context before and after token
        before = sentence[max(0, index - self.context[0]) : index]
        after  = sentence[index+1 : min(sentence.shape[0], index + self.context[1] + 1)]

        # Add padding if necessary
        before = np.pad(before, (self.context[0] - before.shape[0], 0), constant_values=self.no_token)
        after  = np.pad(after , (0, self.context[1] - after .shape[0]), constant_values=self.no_token)

        # Merge into single context and return
        return np.concatenate((before, after))


    def _get_context_full_document(self, token: Token) -> np.ndarray:
        """Compute the sentence context for a token in the document.
            Used when self.sentence_only == False.

            Parameters
            ----------
            token : spacy.tokens.Token
                Token for which to generate the context.

            Returns
            -------
            context : np.array of shape=(_context_before + _context_after)
                Context for token in the document. Concatenation of
                context_before and context_after tokens. To get the
                context_before and context_after separately, simply call
                context[:, :context_before] or context[:, -context_after:]
                respectively.
            """
        # Get document of token
        document = token.doc

        # Get token index
        index = token.i

        # Get context before and after token
        before = np.asarray(
            document[max(0, index - self.context[0]) : index],
            dtype = object,
        )

        after = np.asarray(
            document[index+1 : min(len(document), index + self.context[1] + 1)],
            dtype = object,
        )

        # Add padding if necessary
        before = np.pad(before, (self.context[0] - before.shape[0], 0), constant_values=self.no_token)
        after  = np.pad(after , (0, self.context[1] - after .shape[0]), constant_values=self.no_token)

        # Merge into single context and return
        return np.concatenate((before, after))


    def _get_context_doc_full_document(self, document: Doc) -> np.ndarray:
        """Retrieve the token contexts for all tokens in document as np array.
            Implementation for when self.sentence_only = False

            Parameters
            ----------
            document : spacy.tokens.Doc
                Document for which to retrieve the contexts.

            Returns
            -------
            context : np.array of shape=(n_tokens, _context_before + _context_after)
                Context for tokens in the document. Concatenation of
                context_before and context_after tokens. To get the
                context_before and context_after separately, simply call
                context[:, :context_before] or context[:, -context_after:]
                respectively.
            """
        raise NotImplementedError(
            "get_context_doc() is only implemented for sentence_only = True"
        )


    def _get_context_doc_sentence(self, document: Doc) -> np.ndarray:
        """Retrieve the token contexts for all tokens in document as np array.
            Implementation for when self.sentence_only = True

            Parameters
            ----------
            document : spacy.tokens.Doc
                Document for which to retrieve the contexts.

            Returns
            -------
            context : np.array of shape=(n_tokens, _context_before + _context_after)
                Context for tokens in the document. Concatenation of
                context_before and context_after tokens. To get the
                context_before and context_after separately, simply call
                context[:, :context_before] or context[:, -context_after:]
                respectively.
            """
        # Initialise result
        result = list()

        # Loop over all sentences
        for sentence in document.sents:
            # Initialise context
            before = np.full((len(sentence), self.context[0]), self.no_token)
            after  = np.full((len(sentence), self.context[1]), self.no_token)

            # Create context
            for index in range(self.context[0]):
                print(sentence[index:])
                # print(sentence[:-self.context[0]-index])

            exit()
            for index in range(self.context[1]):
                print(index)

            exit()