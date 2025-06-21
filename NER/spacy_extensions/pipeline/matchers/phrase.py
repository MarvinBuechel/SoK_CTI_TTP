import spacy
from spacy.pipeline import EntityRuler
from spacy.pipeline.entityruler import DEFAULT_ENT_ID_SEP, entity_ruler_score
from spacy.tokens import Doc, Span, Token
from typing import Callable, Iterable, List, Optional, Union

from spacy_extensions.utils.hash_dict import HashableDict

@spacy.Language.factory('matcher_phrases')
class MatcherPhrases(EntityRuler):
    """Intelligent entity matcher for phrases.
        Instead of looking for exact phrase matches, this matcher also accepts
        matches for phrases with various different spellings, suffixes, and
        other common slight variants of phrases.

        Note
        ----
        When a text matches multiple rules, the ``ent_id_sep`` value is used to
        split the labels. E.g., ``Testing -> VERB``, and``Test -> AUTOMATION``
        will both match the text ``testing``, which will therefore get the ENT
        value of ``VERB||AUTOMATION``, where in this case ``ent_id_sep == ||``.

        Requires
        --------
        matcher_dict : Token
            Matcher dict for matching token, provided by spacy_extensions.
            pipeline.TokenBase.

        Extensions
        ----------
        phrases : Doc
            Loop over all matching phrases found in document.
            See :py:meth:`phrases`
    """

    def __init__(
            self,
            nlp,
            name,
            phrase_matcher_attr: Optional[Union[int, str]] = None,
            validate           : bool = False,
            overwrite_ents     : bool = False,
            ent_id_sep         : str = DEFAULT_ENT_ID_SEP,
            scorer             : Optional[Callable] = None,
        ):
        """MatcherPhrases constructor.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which MatcherRegex is used.

            name : string
                Name of pipeline.

            Config Parameters
            -----------------
            See `EntityRuler <https://spacy.io/api/entityruler>`_
            """
        # Initialise super
        super().__init__(
            nlp                 = nlp,
            name                = name,
            phrase_matcher_attr = phrase_matcher_attr,
            validate            = validate,
            overwrite_ents      = overwrite_ents,
            ent_id_sep          = ent_id_sep,
            scorer              = scorer or entity_ruler_score,
        )

        # Register extensions
        Doc.set_extension(
            name   = 'phrases',
            getter = self.phrases,
        )

    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def phrases(self, doc: Doc) -> Iterable[Span]:
        """Iterate over all detected phrases in document.
        
            Parameters
            ----------
            doc : Doc
                Document in which to find phrases.
                
            Yields
            ------
            phrase : Span
                Detected phrase in document.
            """
        raise NotImplementedError("phrases not yet implemented")

    ########################################################################
    #                             Add patterns                             #
    ########################################################################

    def add_phrases(self, X: Iterable[str], y: Iterable[str]) -> None:
        """Transform phrases into patterns based on intelligent pattern matching
            and add these patterns to the underlying EntityRuler.

            Note
            ----
            This method can be used instead of using the 
            ``EntityRuler.add_patterns`` method. If you want more control over
            your patterns, you can use the :py:meth:`string2pattern` method to
            transform strings into patterns and add these resulting patterns as
            strings using the ``EntityRuler.add_patterns`` method:

            .. code::

                # Where we assume that the MatcherPhrase object is called
                # matcher_phrase.
                matcher_phrase.add_patterns([
                    {
                        'pattern': matcher_phrase.string2pattern("Example string"),
                        'label'  : 'example_label', 
                    },
                ])
        
            Parameters
            ----------
            X : Iterable[str] of size=(n_patterns,)
                Patterns on which to match.

            y : Iterable[str] of size=(n_patterns,)
                Labels on which to match.
            """
        # Create list of patterns
        mapping = dict()
        for pattern, label in zip(X, y):
            pattern = self.string2pattern(pattern)
            key = tuple(map(HashableDict, pattern))
            
            if key not in mapping:
                mapping[key] = {
                    'pattern': pattern,
                    'labels' : set(),
                }
            mapping[key]['labels'].add(label)

        # Create unique key for each value
        super().add_patterns([{
                'label': self.ent_id_sep.join(sorted(value['labels'])),
                'pattern': value['pattern'],
            } for _, value in mapping.items()
        ])

    ########################################################################
    #                       Create matcher patterns                        #
    ########################################################################

    def string2pattern(self, string: str) -> List[dict]:
        """Create a Matcher pattern from a given string.
        
            Parameters
            ----------
            string : str
                String from which to generate a matcher pattern.

            Returns
            -------
            pattern : List[dict]
                Resulting matcher pattern.
            """
        # Initialise list
        pattern = list()

        # Disable pipe, if required
        enabled = self.name in dict(self.nlp.pipeline)
        if enabled: self.nlp.disable_pipe(self.name)

        # Loop over all tokens in string
        for token in self.nlp(string):
            # Add any non-punct token
            if not token.is_punct:
                # Return base form of token
                pattern.append(token._.matcher_dict)

                # Allow for arbitrary, weird punctuation in between
                pattern.append({
                    'IS_PUNCT': True,
                    'OP': '*',
                })

        # Re-enable pipe
        if enabled: self.nlp.enable_pipe(self.name)

        # Return pattern
        return pattern[:-1]
