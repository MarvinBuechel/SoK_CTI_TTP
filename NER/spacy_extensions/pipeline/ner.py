# Imports
import inspect
import json
import spacy
from collections import OrderedDict
from importlib import import_module
from itertools import chain
from spacy.tokens import Span, Doc
from spacy_extensions.utils import SpacyCallable, SpacySerializable
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
from typing import OrderedDict as OrdDict


# Custom types
MatcherType = Callable[
    [Union[Doc, Span], bool],
    Union[List[Tuple[int, int, int]], List[Span]]
]


@spacy.Language.factory("ner_matcher")
class NERMatcher(SpacyCallable, SpacySerializable):
    """Create a pipeline that performs named entity recognition (NER) for
        configured matchers."""
    
    def __init__(
            self,
            nlp,
            name,
            extension: str = 'ner_matches',
            force: bool = True,
        ):
        """Configure NERMatcher.
        
            Parameters
            ----------
            nlp : spacy.language.Language
                Pipeline in which NERMatcher operates.
                
            name : str
                Name of NERMatcher component.
                
            extension : str, default='ner_matches'
                Extension name set by NERMatcher.
                
            force : bool, default=True
                Can be set to False so not override existing extension.
            """
        self.nlp = nlp
        self.name = name
        self._matchers : OrdDict[str, MatcherType] = OrderedDict()
        self._extension = extension

        # Set extension if not yet exists
        if force or not Doc.has_extension(extension):
            Doc.set_extension(
                name = extension,
                default = None,
                force = force,
            )


    def __call__(self, doc: Doc) -> Doc:
        """Perform NER based on configured matchers.
        
            Parameters
            ----------
            doc : Doc
                Document on which to perform NER.
                
            Returns
            -------
            doc : Doc
                Document where entities are set.
            """
        # Find entities in doc if not yet set
        if getattr(doc._, self._extension) is None:
            ents = [{
                    'label': span.label_,
                    'start': span.start,
                    'end': span.end,
                    'start_char': span.start_char,
                    'end_char': span.end_char,
                } for span in chain.from_iterable(self.match_ents(doc))
            ]
            setattr(doc._, self._extension, ents)

        # Parse ents
        ents = [Span(
                doc = doc,
                start = ent['start'],
                end = ent['end'],
                label = ent['label'],
            ) for ent in getattr(doc._, self._extension)
        ]
        
        # Add existing entities
        ents.extend(doc.ents)

        # Filter entities
        ents = spacy.util.filter_spans(ents)

        # Set document entities
        doc.set_ents(ents)

        # Return document
        return doc

    def __len__(self) -> int:
        """Number of matchers added to NERMatcher.
        
            Returns
            -------
            length : int
                Number of matchers.
            """
        return len(self._matchers)

    def __contains__(self, key: str) -> bool:
        """Check whether the matcher contains rules for a match ID.
        
            Parameters
            ----------
            key : str
                The matcher ID.
            
            Returns
            -------
            result : bool
                Whether the matcher is included in NERMatcher.
            """
        return key in self._matchers

    def add(self, key: str, matcher: MatcherType) -> None:
        """Add matcher to NERMatcher.
        
            Parameters
            ----------
            key : str
                Key by which to identify matcher.
                
            matcher : Matcher
                Matcher to add.
            """
        if key in self:
            raise ValueError(f"Matcher '{key}' already exists.")
        if matcher.vocab != self.nlp.vocab:
            raise ValueError("Vocab not shared between matcher and NERMatcher.")
        self._matchers[key] = matcher

    def remove(self, key: str) -> None:
        """Remove a matcher from the NERMatcher.
            A `KeyError` is raised if the matcher does not exist.
            
            Parameters
            ----------
            key : str
                The ID of the matcher.
            """
        del self._matchers[key]

    def get(self, key: str, default: Optional[Any]=None) -> MatcherType:
        """Retrieve the matcher stored for a key.
        
            Parameters
            ----------
            key : str
                The ID of the matcher.

            default : Optional[Any], default=None
                Default to return if no matcher was found.
                
            Returns
            -------
            matcher : Matcher
                The matcher.
            """
        return self._matchers.get(key, default)

    def match_ents(self, doc: Doc) -> List[List[Span]]:
        """Match all entities using NER entity matching.
        
            Parameters
            ----------
            doc : Doc
                Doc for which to find ents.
                
            Returns
            -------
            ents : List[List[Span]]
                Found entity spans.
            """
        # Find entity matches
        ents = list()
        for matcher in self._matchers.values():
            ents.append(matcher(doc, as_spans=True))

        # Return ents
        return ents

    def to_bytes(self, exclude: Optional[Iterable[str]] = None) -> bytes:
        # Initialise result
        result = OrderedDict()
        # Add individual matcher patterns
        for key, matcher in self._matchers.items():
            # Get key as text
            key = self.nlp.vocab[key].text
            # Get patterns for custom Matchers
            if hasattr(matcher, 'patterns'):
                patterns = matcher.patterns
            # Get patterns for SpaCy Matchers
            else:
                patterns = {
                    self.nlp.vocab[key].text : value
                    for key, value in matcher._patterns.items()
                }

            # Get signature of matcher
            if isinstance(matcher, spacy.matcher.Matcher):
                parameters = []
            else:
                parameters = inspect.signature(type(matcher)).parameters

            # Add matcher
            result[key] = {
                '__module__': type(matcher).__module__,
                '__name__': type(matcher).__name__,
                'configuration': {
                    param: getattr(matcher, param)
                    for param in parameters
                    if hasattr(matcher, param) and param != 'vocab'
                },
                'patterns': patterns,
            }
        # Return result as dict
        return json.dumps(result).encode('utf-8')

    def from_bytes(
            self,
            data: bytes,
            exclude: Optional[Iterable[str]] = None,
        ) -> 'NERMatcher':
        # Decode data from json
        data = json.loads(data.decode('utf-8'), object_pairs_hook=OrderedDict)

        # Recreate matchers
        for key, matcher_config in data.items():
            # Import relevant matcher
            module = import_module(matcher_config['__module__'])
            matcher_cls = getattr(module, matcher_config['__name__'])
            # Construct matcher
            matcher = matcher_cls(
                self.nlp.vocab,
                **matcher_config['configuration'],
            )
            # Add patterns to matcher
            for match_id, pattern in matcher_config['patterns'].items():
                matcher.add(match_id, pattern)
            # Add matcher to object
            self.add(key, matcher)

        # Return self
        return self
