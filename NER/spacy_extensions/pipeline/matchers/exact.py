import spacy
from spacy.pipeline import EntityRuler
from spacy.pipeline.entityruler import DEFAULT_ENT_ID_SEP, entity_ruler_score
from typing import Callable, Iterable, List, Optional, Union
from spacy_extensions.utils.hash_dict import HashableDict
from spacy_extensions.utils.matcher import AttackMixin

@spacy.Language.factory('matcher_exact')
class MatcherExact(EntityRuler, AttackMixin):
    """Find entities based on on exact matching phrases.

        Note
        ----
        When a text matches multiple rules, the ``ent_id_sep`` value is used to
        split the labels. E.g., ``Testing -> VERB``, and``Test -> AUTOMATION``
        will both match the text ``testing``, which will therefore get the ENT
        value of ``VERB||AUTOMATION``, where in this case ``ent_id_sep == ||``.
    """

    def __init__(
            self,
            nlp,
            name,
            ignore_case        : bool = True,
            phrase_matcher_attr: Optional[Union[int, str]] = None,
            validate           : bool = False,
            overwrite_ents     : bool = False,
            ent_id_sep         : str = DEFAULT_ENT_ID_SEP,
            scorer             : Optional[Callable] = None,
        ):
        """MatcherExact constructor.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which MatcherExact is used.

            name : string
                Name of pipeline.

            Config Parameters
            -----------------
            ignore_case : bool, default=True
                If true, ignore case of matches.

            other :
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

        # Set ignore case
        self.ignore_case = ignore_case

    ########################################################################
    #                             Add patterns                             #
    ########################################################################

    def add_phrases(self, X: Iterable[str], y: Iterable[str]) -> None:
        """Transform phrases into patterns based on exact pattern matching and
            add these patterns to the underlying EntityRuler.

            Note
            ----
            This method can be used instead of using the 
            ``EntityRuler.add_patterns`` method. If you want more control over
            your patterns, you can use the :py:meth:`string2pattern` method to
            transform strings into patterns and add these resulting patterns as
            strings using the ``EntityRuler.add_patterns`` method:

            .. code::

                # Where we assume that the MatcherExact object is called
                # matcher_exact.
                matcher_exact.add_patterns([
                    {
                        'pattern': matcher_exact.string2pattern("Example string"),
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

        # Loop over all tokens in string
        for token in self.nlp.make_doc(string):
            if self.ignore_case:
                pattern.append({'LOWER': token.lower_})
            else:
                pattern.append({'TEXT': token.text})

        # Return pattern
        return pattern
