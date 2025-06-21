# Imports
from py_attack import ATTACK, DomainTypes
from typing import Iterable, List, Optional, Tuple

class AttackMixin:
    """A class that adds from_cti and from_attack methods to Matcher classes.
        Requires Matcher class to contain ``add_phrases`` method. See
        :py:meth:`spacy_extensions.pipeline.matchers.phrase.PhraseMatcher`.
        """

    def from_attack(self, attack: ATTACK) -> 'AttackMixin':
        """Load phrases from given MITRE ATT&CK framework.

            Note
            ----
            This method loads phrases from a given MITRE ATT&CK framework. If
            you instead want to load phrases from a path or url, please use the
            :py:meth`from_cti` method instead.
        
            Parameters
            ----------
            attack : ATTACK
                Attack framework from which to load phrases.

            Returns
            -------
            self : self
                Returns self
        """
        # Load ATT&CK framework
        self.attack = attack

        # Create patterns from MITRE ATT&CK framework and add them
        self.add_phrases(*zip(*attack2phrases(attack)))

        # Return self
        return self


    def from_cti(
            self,
            path: Optional[str] = None,
            url : Optional[str] = None,
            domains: List[DomainTypes] = ['enterprise'],
        ) -> 'AttackMixin':
        """Load MatcherMitreAttack using an attack framework from either a given
            ``path`` or ``url``.

            Note
            ----
            This method loads phrases from a given ``path`` or ``url`` pointing
            to a MITRE ATT&CK framework. If you instead want to load phrases
            from a ATTACK object directly, use the :py:meth`from_attack` method
            instead.
    
            Parameters
            ----------
            path : Optional[str]
                Path from which to load ATTACK framework.
                If this is given, it will always prefer to load via path over URL.
                
            url : Optional[str]
                URL from which to load ATTACK framework.
                If this is given, it will always prefer to load via path over URL.
                
            domains : List[DomainTypes], default=['enterprise']
                ATTACK domains to include when loading.

            Returns
            -------
            self : self
                Returns self
            """
        # Load ATT&CK framework from path, if given
        if path is not None:
            attack = ATTACK.load(
                path    = path,
                domains = domains,
            )

        # Download ATT&CK framework from URL, if given
        elif url is not None:
            attack = ATTACK.download(
                url     = path,
                domains = domains,
            )

        # Otherwise, throw error
        else:
            raise ValueError(
                "Please configure either a path or url for the MITRE ATT&CK CTI "
                "framework to use in the AttackMixin class."
            )

        # Load self from ATT&CK
        return self.from_attack(attack)

################################################################################
#                            Load ATT&CK framework                             #
################################################################################

def attack2phrases(attack: ATTACK) -> Iterable[Tuple[str, str]]:
    """Extract phrases and labels describing each ATT&CK concept from the given
        ATTACK framework.
    
        Parameters
        ----------
        attack : ATTACK
            MITRE ATT&CK framework from which to extract phrases and labels.

        Yields
        ------
        X : str
            Phrase pattern (str) on which to match.
            Can be used as input for ``MatcherPhrases.add_phrases``.

        y : str
            Label corresponding to phrase pattern on which to match.
            Can be used as input for ``MatcherPhrases.add_phrases``.
        """
    # Filter relevant concepts
    concepts = filter(
        lambda concept: (concept.get('identifier') or '\0')[0] in {
            'T', # Tactics and techniques
            'S', # Software
            'G', # Groups
            'M', # Mitigations
            'D', # Data sources
        },
        attack.concepts
    )

    # Loop over all concepts
    for concept in concepts:
        # Get concept label and name
        label = concept.get('identifier')
        name  = concept.get('name')
        # Get all aliases
        aliases = set(
            [name, label] +
            concept.get('aliases', list()) +
            concept.get('x_mitre_aliases', list())
        )

        # Loop over each alias as if it were a Doc
        for alias in aliases:
            # Skip special case 'at'
            if alias == 'at' and label == 'S0110': continue

            # Yield labels and aliases
            yield alias, label
