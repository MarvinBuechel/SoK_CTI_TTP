# Imports
from collections            import OrderedDict
from spacy_extensions.utils import SpacyCallable, SpacySerializable
from spacy_extensions.utils.iocs import iocs2json, json2iocs
from typing                 import Dict, Iterable, Literal, Optional, Union
import spacy
import re

@spacy.Language.factory('matcher_regex')
class MatcherRegex(SpacyCallable, SpacySerializable):
    """MatcherRegex for creating spans and ents from regex definitions."""

    ########################################################################
    #                                Setup                                 #
    ########################################################################

    def __init__(
            self,
            nlp,
            name,
            alignment_mode: Literal['strict', 'contract', 'expand'] = 'strict',
        ):
        """MatcherRegex for creating spans from regex definitions.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which MatcherRegex is used.

            name : string
                Name of pipeline.

            Config Parameters
            -----------------
            alignment_mode : Literal['strict', 'contract', 'expand'], default="strict"
                See alignment_mode of https://spacy.io/api/doc#char_span
            """
        # Set document
        self.nlp  = nlp
        self.name = name

        # Configuration
        self.alignment_mode  = alignment_mode

        # Set regexes
        self.regexes = OrderedDict()

    ########################################################################
    #                                 Call                                 #
    ########################################################################

    def __call__(self, document):
        """Identify spans from registered regexes and add labels to configured
            attribute.

            Parameters
            ----------
            document : spacy.tokens.doc.Doc
                Document in which to identify spans.

            Returns
            -------
            document : spacy.tokens.doc.Doc
                Document where spans are identified based on registered regexes
                and configured self.mode.
            """
        # Collect all spans
        spans = list()

        # Loop over all regexes
        for label, regex in self.regexes.items():
            # Scan text for regex
            for match in regex.finditer(document.text):
                # Get start and end values
                start, end = match.span()

                # Add corresponding span in document
                spans.append(document.char_span(
                    start, end,
                    label          = label,
                    alignment_mode = self.alignment_mode,
                ))

        # Set document entities
        document.set_ents(
            spacy.util.filter_spans(
                tuple(filter(None, spans)) +
                document.ents
            )
        )

        # Return document
        return document

    ########################################################################
    #                           Register methods                           #
    ########################################################################

    def add_regexes(self, regexes: Dict[str, re.Pattern]) -> None:
        """Add multiple regular expressions to matcher.
        
            Parameters
            ----------
            regexes : Dict[str, re.Pattern]
                Regular expression dictionary of ``name`` -> ``regex`` to add.
            """
        # Loop over all regexes
        for name, regex in regexes.items():
            # Add regex
            self.add_regex(name, regex)


    def add_regex(self, name, regex):
        """Register a regex to be detected in documents.

            Parameters
            ----------
            name : string
                Label to give to span when it is detected from the regex.

            regex : string or re.Pattern
                String to be compiled to regex pattern, or precompiled regex
                pattern to detect in documents.
            """
        # Compile regex if not yet compiled
        if isinstance(regex, str):
            regex = re.compile(regex)

        # Check if name is already registered
        if name in self.regexes and self.regexes[name] != regex:
            raise ValueError(
                f"'{name}' already registered with regex {self.regexes[name]}"
            )

        # Check if regex is of correct type
        if not (
                hasattr(regex, 'finditer') and
                callable(getattr(regex, 'finditer'))
            ):
            raise ValueError(
                f"Unknown regex type '{type(regex)}', please provide an "
                "re.Pattern, string or custom class that implements the "
                "'finditer' method."
            )

        # Register regex
        self.regexes[name] = regex

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_bytes(self, exclude: Optional[Iterable[str]] = None) -> bytes:
        """Serialize object to bytes.
        
            Parameters
            ----------
            exclude : Optional[Iterable[str]], default=None
                Parameters to exclude.
            """
        # Check if we should serialize regexes
        if exclude is None or 'regexes' not in exclude:
            # Encode regexes as as json and transform to bytes
            return iocs2json(self.regexes).encode('utf-8')

        # Return default
        return b""

    def from_bytes(self, data: bytes, exclude: Optional[Iterable[str]] = None):
        """Serialize object to bytes.
        
            Parameters
            ----------
            exclude : Optional[Iterable[str]], default=None
                Parameters to exclude.

            Returns
            -------
            self : self
                Returns self
            """
        # Check if we should serialize regexes
        if exclude is None or 'regexes' not in exclude:
            # Add regexes from decoded json file
            self.add_regexes(json2iocs(data.decode('utf-8')))

        # Return self
        return self
