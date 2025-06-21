# Imports
from ast import literal_eval
from functools              import partial
from spacy                  import Language
from spacy.tokens           import Token
from spacy_extensions.vocab import file2vocab, filter_vocab, remove_counts, vocab2counter, token2str
from spacy_extensions.utils import SpacyCallable, SpacySerializable
from typing                 import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np

@Language.factory('vocab_mapper')
class VocabMapper(SpacyCallable, SpacySerializable):
    """VocabMapper for mapping tokens to an index used for word Embedders.
        Word embedding methods map a token to a numerical value that can be
        used in neural networks. The VocabMapper assigns a unique identifier
        to each token in the text.

        Note
        ----
        **Important:** make sure that you add a vocabulary to the
        vocab_mapper before using the extensions. You can use the
        :py:meth:`set_vocab`, :py:meth:`set_vocab_file`, or
        :py:meth:`from_disk` methods to load a vocabulary.
        
        Extensions
        ----------
        norm : Token
            The normalized form of a token that is used to map a token to the
            vocabulary.
            
        mapped_index : Token
            The index of a token in the vocabulary. Can be used by other pipes
            as an input for a classifier or neural network.
            
        mapped_vocab : Token
            The vocab entry a token is mapped to.

        mapped_context : Token
            The indices of a token context in the vocabulary.Can be used by
            other pipes as an input for a classifier or neural network.
        """

    def __init__(
            self,
            nlp,
            name: str,
            unk_token : str = '[UNK]',
            sep_token : str = '[SEP]',
            pad_token : str = '[PAD]',
            cls_token : str = '[CLS]',
            mask_token: str = '[MASK]',
            special   : bool = True,
        ):
        """VocabMapper for mapping tokens to an index used for word Embedders.
            Word embedding methods map a token to a numerical value that can be
            used in neural networks. The VocabMapper assigns a unique identifier
            to each token in the text.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which VocabMapper is used.

            name : string
                Name of pipeline.

            Configuration
            -------------
            unk_token : str, default = '[UNK]'
                Token string indicating an unknown value.

            sep_token : str, default = '[SEP]'
                Token string indicating a sentence separation value.

            pad_token : str, default = '[PAD]'
                Token string indicating a padding value.

            cls_token : str, default = '[CLS]'
                Token string indicating a classification token value.

            mask_token : str, default = '[MASK]'
                Token string indicating a masked token value.

            special : bool, default=False
                If True, map IOCs and ATTACK concepts to special tokens.
            """
        # Initialize pipeline
        self.nlp  = nlp
        self.name = name

        # Initialise default tokens
        self.unk_token  = unk_token
        self.sep_token  = sep_token
        self.pad_token  = pad_token
        self.cls_token  = cls_token
        self.mask_token = mask_token
        self.special    = special

        # Register token extensions
        Token.set_extension(
            name   = "norm",
            getter = partial(token2str, special=self.special),
        )

        Token.set_extension(
            name   = "mapped_index",
            getter = self.get_index,
        )

        Token.set_extension(
            name   = "mapped_vocab",
            getter = self.get_vocab,
        )

        Token.set_extension(
            name   = "mapped_context",
            getter = self.get_context,
        )

    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def get_index(self, token: Token) -> int:
        """Returns the index of the token in the mapping."""
        return self.word2index.get(
            token._.norm,
            self.word2index[self.unk_token],
        )

    def get_vocab(self, token: Token) -> str:
        """Returns the mapped str of the token in the vocab."""
        return self.index2word[token._.mapped_index]

    def get_context(self, token: Token) -> Callable:
        """Returns a function which allows you to map the context of a specific
            token. See :py:meth:`context2indices`."""
        return self.context2indices

    ########################################################################
    #                                 TODO                                 #
    ########################################################################

    def context2indices(self, context: Iterable[Optional[Token]]) -> List[int]:
        """Transform an iterable context into a list of mapped indices.
        
            Parameters
            ----------
            context : Iterable[Optional[Token]]
                Context for which to retrieve indices.
                
            Returns
            -------
            result : List[int]
                Mapped indices of tokens.
            """
        # Initialise result
        result = list()

        # Loop over tokens in context
        for token in context:
            # Get index of mapped token
            if isinstance(token, Token):
                result.append(token._.mapped_index)
            # In case of None, we assume padding
            elif token is None:
                result.append(self.pad_index)
            # Exception to catch other token types
            else:
                raise ValueError(
                    f"Cannot retrieve context index: Unknown type {type(token)}"
                )

        # Return result
        return result

    ########################################################################
    #                              Properties                              #
    ########################################################################

    def __len__(self) -> int:
        """Returns size of vocabulary."""
        return len(self.vocab)

    @property
    def unk_index(self):
        """Return index corresponding to unk_token."""
        return self.word2index[self.unk_token]

    @property
    def sep_index(self):
        """Return index corresponding to sep_token."""
        return self.word2index[self.sep_token]

    @property
    def pad_index(self):
        """Return index corresponding to pad_token."""
        return self.word2index[self.pad_token]

    @property
    def cls_index(self):
        """Return index corresponding to cls_token."""
        return self.word2index[self.cls_token]

    @property
    def mask_index(self):
        """Return index corresponding to mask_token."""
        return self.word2index[self.mask_token]

    ########################################################################
    #                            Register vocab                            #
    ########################################################################

    def set_vocab_file(
            self,
            path: Optional[str] = None,
            escape: bool = True,
            topk: Optional[int] = None,
            frequency: int = 0,
        ) -> None:
        """Set vocabulary to vocab_mapper by loading vocab from file.
        
            Parameters
            ----------
            path : Optional[str]
                If given, load vocab from file.

            escape : bool, default=True
                If True, read vocab as if it were escaped.

            topk : Optional[int], default=None
                If given, only use topk most common values in vocab.
                Requires the count for each item to be present in vocab.

            frequency : int, default=None
                Minimum required frequency of item to be in vocab.
                If > 1, requires the count for each item to be present in vocab.
            """
        # Load vocab from file
        self.set_vocab(
            vocab     = file2vocab(path, escape=escape),
            topk      = topk,
            frequency = frequency,
        )


    def set_vocab(
            self,
            vocab: List[Union[str, Tuple[str, int]]],
            topk: Optional[int] = None,
            frequency: int = 0,
        ) -> None:
        """Set vocabulary to vocab_mapper.
        
            Parameters
            ----------
            vocab : List[Union[str, Tuple[str, int]]]
                Vocabulary to use for mapping.

            topk : Optional[int], default=None
                If given, only use topk most common values in vocab.
                Requires the count for each item to be present in vocab.

            frequency : int, default=None
                Minimum required frequency of item to be in vocab.
                If > 1, requires the count for each item to be present in vocab.
            """
        # Configure vocab
        self.vocab = remove_counts(filter_vocab(
            vocab      = vocab2counter(vocab),
            topk       = topk,
            frequency  = frequency,
            unk_token  = self.unk_token,
            sep_token  = self.sep_token,
            pad_token  = self.pad_token,
            cls_token  = self.cls_token,
            mask_token = self.mask_token,
        ))

        # Perform checks
        special_tokens = [
            self.unk_token,
            self.sep_token,
            self.pad_token,
            self.cls_token,
            self.mask_token,
        ]
        for special_token in special_tokens:
            if special_token not in self.vocab:
                raise ValueError(f"Special token {special_token} not in vocab.")

        # Set mapping
        self.word2index = dict()
        self.index2word = dict()        
        for index, word in enumerate(self.vocab):
            # Add token to mapping
            self.word2index[word ] = index
            self.index2word[index] = word

    ########################################################################
    #                             I/O methods                              #
    ########################################################################


    def to_bytes(self, exclude: Optional[Iterable[str]] = None) -> bytes:
        """Return bytes representation of vocab_mapper.
        
            Parameters
            ----------
            exclude : Optional[Iterable[str]], default=None
                If 'vocabulary' is specified, do not store vocab.
            """
        # Case if vocab should be excluded
        if exclude and 'vocabulary' in exclude:
            return b""

        # Return token list
        return '\n'.join(repr(token) for token in self.vocab).encode('utf-8')
        
        

    def from_bytes(self, data: bytes, exclude: Optional[Iterable[str]] = None):
        """Load vocab_mapper from bytes representation.
        
            Parameters
            ----------
            exclude : Optional[Iterable[str]], default=None
                If 'vocabulary' is specified, do not store vocab.
            """
        # Case if vocab should be excluded
        if not exclude or 'vocabulary' not in exclude:
            # Read vocabulary
            self.set_vocab([
                literal_eval(token)
                for token in data.decode('utf-8').split('\n')
            ])
        
        return self
