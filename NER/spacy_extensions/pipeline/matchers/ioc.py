# Imports
from functools   import partial
from pathlib import Path
from typing      import Iterable, Literal, Tuple, Union
import re
import spacy
from spacy.tokens import Doc, Span, Token

# Local imports
from spacy_extensions.pipeline.matchers import MatcherRegex
from spacy_extensions.utils.iocs import AT_SEP, DOT_SEP, IPv6SEP, jsonfile2iocs


@spacy.Language.factory('matcher_ioc')
class MatcherIoC(MatcherRegex):
    """Add IOC entity recognition to pipeline via regular expression detection.
        Find spans of indicators of compromise from regex definitions. See
        ``spacy_extensions.utils.iocs.IOCS`` for the currently supported
        indicators of compromise.
        
        Superclasses
        ------------
        MatcherRegex
            The MatcherIoC is a subclass of :py:class:`MatcherRegex`.

        Extensions
        ----------
        iocs : Doc
            Iterator over all found IOCs in document.
            See :py:meth:`filter_ents`.

        ioc_type : Union[Span, Token]
            Retrieves the IOC type (if any) from the given span or token.
            See :py:meth:`ioc_type`.

        refang : Union[Span, Token]
            Removes the defang additions of the found IOC.
            See :py:meth:`refang`.
        """

    def __init__(
            self,
            nlp,
            name,
            alignment_mode: Literal['strict', 'contract', 'expand'] = 'strict',
            force : bool = True,
        ):
        """MatcherIoC for creating spans of IOCs from regex definitions.
        
            Parameters
            ----------
            See :ref:`MatcherRegex`
            """
        # Initialise super
        super().__init__(
            nlp            = nlp,
            name           = name,
            alignment_mode = alignment_mode,
        )

        # Set variables
        self.fangs = {
            '.': re.compile(DOT_SEP),
            '@': re.compile(AT_SEP),
            ':': re.compile(IPv6SEP),
        }

        # Add additional document extensions
        if force or not Doc.has_extension('iocs'):
            Doc.set_extension(
                name = 'iocs',
                getter = partial(self.filter_ents, ent_labels=self.regexes),
                force = force,
            )
        if force or not Doc.has_extension('iocs_all'):
            Doc.set_extension(
                name = 'iocs_all',
                getter = self.iocs_all,
                force = force,
            )

        # Return IOC type if any
        if force or not Span.has_extension('ioc_type'):
            Span.set_extension(
                name = 'ioc_type',
                getter = self.ioc_type,
                force = force,
            )

        if force or not Token.has_extension('ioc_type'):
            Token.set_extension(
                name = 'ioc_type',
                getter = self.ioc_type,
                force = force,
            )

        # Get base (refanged) form of IOC
        if force or not Span.has_extension('refang'):
            Span.set_extension(
                name = 'refang',
                getter = self.refang,
                force = force,
            )

        if force or not Token.has_extension('refang'):
            Token.set_extension(
                name = 'refang',
                getter = self.refang,
                force = force,
            )

    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def filter_ents(self, doc: Doc, ent_labels: Union[str, Iterable[str]]):
        """Retrieve all entities in doc matching the given ent_labels.
        
            Parameters
            ----------
            doc : Doc
                Document for which to retrieve the given entities.
                
            ent_labels : Union[str, Iterable[str]]
                Labels on which to match.
            
            Yields
            ------
            entity : Span
                Entity matching the given ent_labels.
            """
        if isinstance(ent_labels, str):
            yield from filter(
                lambda x: self.nlp.vocab[x.label].text == self.regexes,
                doc.ents,
            )
        else:
            yield from filter(
                lambda x: self.nlp.vocab[x.label].text in self.regexes,
                doc.ents,
            )


    def iocs_all(self, doc: Doc) -> Iterable[Tuple[re.Match, str]]:
        """Retrieve all iocs independent of alignment_mode. This means it will
            return all spans that are found.
            
            Parameters
            ----------
            doc : Doc
                Document in which to find IOCs.
            
            Yields
            ------
            match : re.Match
                Matches in document text.

            label : str
                Label corresponding to IOC.
            """
        # Initialise IOCs
        iocs = list()

        # Loop over all regexes
        for label, regex in self.regexes.items():
            # Scan text for regex
            for match in regex.finditer(doc.text):
                # Add match, label tuple
                iocs.append((match, label))

        # Sort IOCs
        iocs = sorted(iocs, key=lambda x: (x[0].start(), -x[0].end()))

        # Get seen ranges
        seen = list()

        # Filter spans based on longest first
        for match, label in iocs:
            # Get range
            range_ = set(range(*match.span()))

            # Check if range was already seen
            if not any(range_ & r for r in seen):
                # Add to seen
                seen.append(range_)
                # Yield result
                yield match, label
        


    def ioc_type(self, obj: Union[Span, Token]) -> str:
        """Get the IOC type from a given IOC
        
            Parameters
            ----------
            obj : Union[Span, Token]
                SpaCy obj for which to retrieve IOC type.
            
            Returns
            -------
            type : str
                Name of IOC type.
            """
        # Retrieve IOC type
        if isinstance(obj, Span):
            type = self.nlp.vocab[obj.label].text
        else:
            type = obj.ent_type_

        # Fix URL_TLD
        if type == 'URL_TLD': type = "URL"

        # Check if IOC type is of valid type and return
        return type if type in self.regexes else None


    def refang(self, obj: Union[Span, Token]) -> str:
        """Return the refanged form of given IOC.
        
            Parameters
            ----------
            obj : Union[Span, Token]
                IOC for which to get the refanged version.
                
            Returns
            -------
            result : str
                Refanged version of IOC.
            """
        # Initialise result
        result = obj.text

        # Apply refang if result is IOC
        for fang, regex in self.fangs.items():
            result = regex.sub(fang, result)

        # Return result
        return result

    ########################################################################
    #                           Register methods                           #
    ########################################################################

    def from_file(self, path: Union[str, Path]) -> None:
        """Load IOCs from given file. 

            Note
            ----
            This method loads IOCs from a given file. If you instead want to
            load IOCs from a dictionary, please use the :py:meth`add_regexes`
            method instead. This library comes with the default
            ``/config/iocs.json`` file which is a file representation of the
            IOCs from ``spacy_extensions.utils.iocs.IOCs``. This file was
            created using the ``spacy_extensions/utils/iocs.py`` script.
        
            Parameters
            ----------
            path : Union[str, Path]
                Path from which to load IOCs.
        """
        # Load IOCs from path
        self.add_regexes(jsonfile2iocs(path))
