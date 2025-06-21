import io
import json
import pandas as pd
import spacy
from ast import literal_eval
from collections import defaultdict
from functools import partial
from spacy.tokens import Doc, Token
from spacy_extensions.utils import SpacyCallable, SpacySerializable
from tqdm import tqdm
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple

@spacy.Language.factory('related_wordnet',
    assigns  = [
        'token._.related_base',
        'token._.related_lemmas',
        'token._.related_wordnet'
    ],
    requires = ['token._.token_base', 'token._.synsets'],
)
class RelatedWordNet(SpacyCallable, SpacySerializable):
    """Find the trained related wordnet (token, label) tuples for each token."""

    def __init__(self, nlp, name, force: bool = True):
        """RelatedWordNet constructor.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which RelatedTokens is used.

            name : string
                Name of pipeline.

            force : bool, default=True
                If True, overwrite existing extensions.
            """
        # Set pipeline
        self.nlp  = nlp
        self.name = name

        # Set lookup table
        self.lookup = dict()
        self.token_lookup = TrainedTokenLookup(self.nlp)

        # Set extensions
        if force or not Token.has_extension('related_wordnet'):
            Token.set_extension(
                name   = 'related_wordnet',
                getter = self.related,
                force = force,
            )

        # Set extensions
        if force or not Token.has_extension('related_lemmas'):
            Token.set_extension(
                name   = 'related_lemmas',
                getter = partial(
                    self.related_attribute,
                    getter = lambda x: x.lemma_,
                ),
                force = force,
            )
        if force or not Token.has_extension('related_base'):
            Token.set_extension(
                name   = 'related_base',
                getter = partial(
                    self.related_attribute,
                    getter = lambda x: x._.token_base,
                ),
                force = force,
            )


    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def related(self, token: Token) -> Set[Tuple[Token, str]]:
        """Retrieve (token, label) tuples related to the current token 
            according to trained wornet synsets.
            
            Parameters
            ----------
            token : Token
                Token for which to find related tokens.
                
            Returns
            -------
            related : Set[Tuple[Token, str]]
                Trained (token, label) tuples related to current token.
            """
        # Initialise result
        result = set([(token, None)])

        # Add to result
        for synset in token._.synsets:
            # Get all related names
            for item in self.lookup.get(synset.name(), []):
                key = (item['token'], item['label'])
                for related in self.token_lookup.get(key):
                    result.add((related, item['label']))

        # Return result
        return result

    def related_attribute(
            self,
            token: Token,
            getter: Callable[[Token], Any] = lambda x: x.lemma_,
        ) -> Set[str]:
        """Find the attributes of trained related tokens for a given token.
            
            Parameters
            ----------
            token : Token
                Token for which to find related token attributes.

            getter : Callable[[Token], Any], default=lambda x: x.lemma_
                Getter for token attribute, defaults to token lemma.
                
            Returns
            -------
            related : Set[str]
                Related trained attributes for current token.
            """
        return set((
            getter(related) for related, _ in token._.related_wordnet
        ))

    ########################################################################
    #                              Train pipe                              #
    ########################################################################

    def fit(
            self,
            synsets: Iterable[List[str]],
            tokens: Iterable[str],
            labels: Iterable[str],
            examples: Iterable[str],
            token_lookup: Optional['TrainedTokenLookup'],
        ) -> 'RelatedWordNet':
        """Fit related items in wordnet with labelled synsets.
        
            Parameters
            ----------
            synsets : Iterable[List[str]]
                List of synsets corresponding to each token.

            tokens : Iterable[str]
                Tokens corresponding to each list of synsets.
            
            labels : Optional[Iterable[str]]
                Optional label corresponding to synset.

            Returns
            -------
            self : self
                Returns self
            """
        # Clear lookup table
        self.lookup = defaultdict(list)

        # Loop over all 
        for synsets_, token, label in zip(synsets, tokens, labels):
            for synset in synsets_:
                self.lookup[synset].append({
                    'token': token,
                    'label': label,
                })

        # Set token lookup
        if token_lookup is None:
            lab, ex = (zip(*set(zip(labels, examples))))
            assert len(lab) == len(set(lab))
            self.token_lookup = TrainedTokenLookup(self.nlp).fit(ex, lab)
        else:
            self.token_lookup = token_lookup

        # Return self
        return self

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_bytes(self, exclude: Optional[Iterable[str]] = None) -> bytes:
        """Get byte representation from object.
        
            Parameters
            ----------
            exclude : Optional[Iterable[str]]
                Ignored.

            Returns
            -------
            result : bytes
                Bytes representation of object.
            """
        # Initialise temporary representation
        representation = defaultdict(list)

        # Loop over lookup table
        for synset, related in self.lookup.items():
            for item in related:
                representation[(
                    item['label'],
                    item['token'],
                )].append(synset)

        # Transform representation into dataframe
        df = pd.DataFrame([{
                'identifier': identifier,
                'token'     : token,
                'synsets'   : list(sorted(synsets)),
                'name'      : self.token_lookup.examples[identifier].text,
            } for (identifier, token), synsets in representation.items()
        ])

        # Write to buffer
        output_buffer = io.StringIO()
        df.to_csv(output_buffer, index=None)

        # Set result
        result = {
            'related': output_buffer.getvalue(),
            'token_lookup': self.token_lookup.to_bytes().decode('utf-8'),
        }

        # Return as bytes
        return json.dumps(result).encode('utf-8')

        # Return as bytes
        return output_buffer.getvalue().encode('utf-8')


    def from_bytes(
            self,
            data: bytes,
            exclude: Optional[Iterable[str]] = None,
        ) -> SpacySerializable:
        """Load object from its byte representation.
        
            Parameters
            ----------
            data : bytes
                Data from which to load object.

            exclude : Optional[Iterable[str]]
                Ignored.

            Returns
            -------
            self : self
                Returns self.
            """
        # Load bytes
        try:
            data = json.loads(data.decode('utf-8'))
            token_lookup = TrainedTokenLookup(self.nlp).from_bytes(
                data['token_lookup'].encode('utf-8')
            )
            data = data['related']
        except Exception as e:
            import warnings
            warnings.warn(f"Depricated from_bytes: {e}")
            data = data.decode('utf-8')
            token_lookup = None

        # Load csv data into dataframe
        try:
            df = pd.read_csv(io.StringIO(data))
        except pd.errors.EmptyDataError as e:
            return self

        # Parse synsets
        df['synsets'] = [literal_eval(synset) for synset in df['synsets']]

        # Skip empty synsets
        df = df[[len(synset) > 0 for synset in df['synsets']]]

        # Fit self and return
        return self.fit(
            synsets  = df['synsets'].values,
            tokens   = df['token'].values,
            labels   = df['identifier'].values,
            examples = df['name'].values,
            token_lookup = token_lookup,
        )


class TrainedTokenLookup(SpacySerializable):

    def __init__(self, nlp: spacy.language.Language):
        """Initialise TrainedTokenLookup"""
        # Initialise NLP and examples
        self.nlp = nlp
        self.examples = dict()

    ########################################################################
    #                             Fit/Predict                              #
    ########################################################################

    def fit(
            self,
            examples: Iterable[str],
            labels: Iterable[str],
            required: Set[str] = {
                'tok2vec', 'transformer', 'tagger', 'parser',
                'attribute_ruler', 'lemmatizer',
            },
            verbose: bool = False,
        ) -> 'TrainedTokenLookup':
        """Train TrainedTokenLookup with given examples and labels.
        
            Parameters
            ----------
            examples : Iterable[str]
                Example strings to train with.
            
            labels : Iterable[str]
                Labels related to each given example.
            
            required : Set[str]
                Required pipes to parse labels.
            
            Returns
            -------
            self : self
                Returns self.
            """
        # Disable all but required pipes in nlp
        disabled = set()
        for name, _ in self.nlp.pipeline:
            if name not in required:
                disabled.add(name)
                self.nlp.disable_pipe(name)

        # Create iterator
        iterator = zip(self.nlp.pipe(examples), labels)
        # Add verbosity, if necessary
        if verbose: iterator = tqdm(iterator, desc='Train token lookup')

        # Create examples
        self.examples = {label: ex for ex, label in iterator}

        # Re-enable pipes
        for name in disabled:
            self.nlp.enable_pipe(name)

        # Return self
        return self
            

    def get(self, item: Tuple[str, str]) -> List[Token]:
        """Get matching token(s) from given item."""
        return [t for t in self.examples.get(item[1], []) if t.text == item[0]]

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_bytes(self, exclude: Optional[Iterable[str]] = None) -> bytes:
        """Convert all examples to bytes and return."""
        return json.dumps({
            label: doc.to_json() for label, doc in self.examples.items()
        }).encode('utf-8')

    def from_bytes(
            self,
            data: bytes, exclude: Optional[Iterable[str]] = None
        ) -> SpacySerializable:
        """Load TrainedTokenLookup from bytes."""
        # Load examples
        self.examples = {
            label: Doc(self.nlp.vocab).from_json(doc)
            for label, doc in json.loads(data.decode('utf-8')).items()
        }

        # Return self
        return self