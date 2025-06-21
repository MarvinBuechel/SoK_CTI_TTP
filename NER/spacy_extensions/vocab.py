from ast          import literal_eval
from collections  import Counter
from spacy.tokens import Doc, Token
from tqdm         import tqdm
from typing       import Any, Callable, Iterable, List, Optional, Tuple, Union

################################################################################
#                                 Create vocab                                 #
################################################################################

def create_vocab_file(
        outfile      : str,
        documents    : Iterable[Doc],
        include_count: bool = True,
        escape       : bool = False,
        *args, **kwargs,
    ) -> None:
    """Create vocab file from given documents.

        Parameters
        ----------
        outfile : str
            File to write vocab to.

        vocab : List[Union[str, Tuple[str, int]]]
            Vocabulary to write.

        include_count : bool, default=True
            If True, include count in tokens.

        escape : bool, default=False
            If True, store vocab as repr(vocab)

        *args : Any
            See `create_vocab`.

        **kwargs : Any
            See `create_vocab`.
        """
    # Create vocabulary
    vocab = create_vocab(documents, *args, **kwargs)

    # Write to output file
    vocab2file(
        outfile       = outfile,
        vocab         = vocab,
        include_count = include_count,
        escape        = escape,
    )


def create_vocab(
        documents : Iterable[Doc],
        topk      : Optional[int] = None,
        frequency : int  = 1,
        unk_token : str  = '[UNK]',
        sep_token : str  = '[SEP]',
        pad_token : str  = '[PAD]',
        cls_token : str  = '[CLS]',
        mask_token: str  = '[MASK]',
        special   : Union[bool, Callable[[Token], str]] = True,
        verbose   : bool = False,
    ) -> List[Tuple[str, int]]:
    """Create a vocab from given documents.

        Parameters
        ----------
        documents : Iterable[Doc]
            Documents used to create the vocabulary.

        topk : int, optional
            If given, only use the topk most common values in vocabulary.

        frequency : int, default=1
            The minimum required frequency of terms to end up in vocabulary.

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

        special : bool, default=True
            If True, map IOCs and ATTACK concepts to special tokens.

        verbose : boolean, default=False
            If True, print progress.

        Returns
        -------
        vocab : List[Tuple[str, int]]
            Vocabulary and count gathered from documents.
        """
    # Initialise tokens
    tokens = list()

    # Add verbosity, if necessary
    if verbose: documents = tqdm(documents, desc='Create vocab')

    # Gather all normalized tokens
    for document in documents:
        for token in document:
            if not token.is_space:
                # If a bool is given, use token2str method
                if isinstance(special, bool):
                    tokens.append(token2str(token, special=special))
                # Otherwise, use callable special method
                else:
                    tokens.append(special(token))

    # Filter vocab and return
    return filter_vocab(
        vocab      = Counter(tokens),
        topk       = topk,
        frequency  = frequency,
        unk_token  = unk_token,
        sep_token  = sep_token,
        pad_token  = pad_token,
        cls_token  = cls_token,
        mask_token = mask_token,
    )


def merge_vocabs(
        vocabs: List[List[Union[str, Tuple[str, int]]]],
        *args: Any,
        **kwargs: Any,
    ) -> List[Union[str, Tuple[str, int]]]:
    """Merge multiple vocabs together into a single vocab.
    
        Parameters
        ----------
        vocabs : List[List[Union[str, Tuple[str, int]]]]
            Vocabularies to merge together.
        
        *args : Any
            See `filter_vocab`
            
        **kwargs : Any
            See `filter_vocab`
        """
    # Merge multiple vocabs together
    vocab = sum([vocab2counter(vocab) for vocab in vocabs], Counter())

    # Filter vocabulary
    return filter_vocab(vocab, *args, **kwargs)


def filter_vocab(
        vocab     : Counter,
        topk      : Optional[int] = None,
        frequency : int           = 1,
        unk_token : str           = '[UNK]',
        sep_token : str           = '[SEP]',
        pad_token : str           = '[PAD]',
        cls_token : str           = '[CLS]',
        mask_token: str           = '[MASK]',
    ) -> List[Tuple[str, int]]:
    # Filter vocabulary
    vocab = list(sorted([
        (token, count) for token, count in vocab.most_common(topk)
        if count >= frequency
    ]))

    # Get vocab as set
    vocab_set = set([token for token, count in vocab])

    # Add special tokens to vocab if not already in
    special_tokens = list()
    if unk_token not in vocab_set:
        special_tokens.append((unk_token, 0))
    if sep_token not in vocab_set:
        special_tokens.append((sep_token, 0))
    if pad_token not in vocab_set:
        special_tokens.append((pad_token, 0))
    if cls_token not in vocab_set:
        special_tokens.append((cls_token, 0))
    if mask_token not in vocab_set:
        special_tokens.append((mask_token, 0))

    # Add special tokens and return
    return special_tokens + vocab


def vocab2counter(vocab: List[Union[str, Tuple[str, int]]]) -> Counter:
    """Transform a vocab to a counter.
    
        Parameters
        ----------
        vocab : List[Union[str, Tuple[str, int]]]
            Vocab to transform to counter object.
        
        Returns
        -------
        counter : Counter[str]
            Counter of vocabulary.
        """
    if isinstance(vocab[0], tuple):
        return Counter(dict(vocab))
    else:
        return Counter(vocab)


def remove_counts(vocab: List[Union[str, Tuple[str, int]]]) -> List[str]:
    """Remove counts from a given vocab.
        Vocabs consisting of (token, count) tuples are transformed to a list of
        tokens. Vocabs consisting of a list of tokens are unmodified.
        
        Parameters
        ----------
        vocab : List[Union[str, Tuple[str, int]]]
            Vocab from which to remove counts.

        Returns
        -------
        vocab : List[str]
            Vocab where counts are removed.
        """
    # Initialise new vocab
    result = list()

    # Remove count per vocab entry
    for entry in vocab:
        if isinstance(entry, tuple):
            result.append(entry[0])
        else:
            result.append(entry)

    # Return result
    return result


################################################################################
#                              Vocab I/O methods                               #
################################################################################

def vocab2file(
        outfile      : str,
        vocab        : List[Union[str, Tuple[str, int]]],
        include_count: bool = True,
        escape       : bool = False,
    ) -> None:
    """Write vocab (e.g., created through `create_vocab` to file.

        Parameters
        ----------
        outfile : str
            File to write vocab to.

        vocab : List[Union[str, Tuple[str, int]]]
            Vocabulary to write.

        include_count : bool, default=True
            If True, include count in tokens.

        escape : bool, default=False
            If True, store vocab as repr(vocab).
        """
    # Set escape function
    escape = lambda x: repr(x) if escape else lambda x: x

    # Transform vocab to strings
    if all(isinstance(token, tuple) for token in vocab):
        if include_count:
            vocab = [f'{count} {escape(token)}' for token, count in vocab]
        else:
            vocab = [escape(token) for token, count in vocab]

    # Check that no entry contains a newline
    for token in vocab:
        if '\n' in token:
            raise ValueError(
                f"Token {repr(token)} contains '\\n' character, "
                 "please set escape=True."
            )

    # Write vocab to file
    with open(outfile, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(vocab) + '\n')


def file2vocab(infile: str, escape: bool = False) -> List[Tuple[str, int]]:
    """Load vocab from file.

        Parameters
        ----------
        infile : str
            Input file from which to read vocab.

        escape : bool, default=False
            If True, read vocab as if it contains escape characters.

        Returns
        -------
        vocab : List[Tuple[str, int]]
            Vocabulary read from file.
        """
    # Initialise vocab
    vocab = list()

    # Read vocab from file
    with open(infile, encoding='utf-8') as infile:
        # Loop over each line in vocab
        for token in infile.readlines():

            # Split token (see if we have a file [count, token] or only tokens)
            token = token.split(' ', 1)
            # [count, token] case
            if len(token) == 2 and isint(token[0]):
                count = int(token[0])
                token = token[1][:-1]
            # Regular tokens
            else:
                count = 0
                token = token[0][:-1]

            # Unescape token if necessary
            if escape:
                token = literal_eval(token)

            # Add to vocab
            vocab.append((token, count))

    # Return vocab
    return vocab


def merge_vocab_files(
        infiles: List[str],
        outfile: str,
        include_count: bool = True,
        escape       : bool = False,
        *args : Any,
        **kwargs : Any,
    ) -> None:
    """Merge multiple vocabulary files into a single file.
    
        Parameters
        ----------
        infiles : List[str]
            Input vocab files to merge.
            
        outfile : str
            Output file where to write resulting vocab.
            
        include_count : bool, default=True
            If True, include the vocab count.

        escape : bool, default=False
            If True, store vocab as repr(vocab).

        *args : Any
            See `filter_vocab`

        **kwargs : Any
            See `filter_vocab`
        """
    # Initialise vocabularies
    vocabs = list()
    
    # Load all given vocabularies
    for infile in infiles:
        vocabs.append(file2vocab(infile, escape))

    # Merge into single vocab
    vocab = merge_vocabs(vocabs, *args, **kwargs)

    # Write vocab to output file
    vocab2file(
        outfile       = outfile,
        vocab         = vocab,
        include_count = include_count,
        escape        = escape,
    )

################################################################################
#                               Token to string                                #
################################################################################

def token2str(token: Token, special: bool = True) -> str:
    """Transform a SpaCy token to its string representation.

        Parameters
        ----------
        token : Token
            Token for which to get string representation.

        special : bool, default=True
            If True, map IOCs and ATTACK concepts to special tokens.

        Returns
        -------
        result : string
            String representation of token.
        """
    # Special tokens for IOC and MITRE ATT&CK types
    if special and token.has_extension('attack_type') and token._.attack_type:
        return f'[{token._.attack_type}]'
    if special and token.has_extension('ioc_type') and token._.ioc_type:
        return f'[{token._.ioc_type}]'

    # Return string representation of token
    return (token.lemma_ or token.norm_).lower()

################################################################################
#                                    Utils                                     #
################################################################################

def isint(string):
    """Check if string is integer"""
    return string.isdigit() or (string.startswith('-') and string[1:].isdigit())
