# Imports
import spacy
from collections import defaultdict
from py_attack import ATTACK
from spacy_extensions.pipeline.matchers.related import MatcherRelated
from spacy_extensions.utils.matcher import AttackMixin, attack2phrases
from tqdm import tqdm

################################################################################
#                                     Pipe                                     #
################################################################################

@spacy.Language.factory('mitre_matcher_related')
class MatcherMitreRelated(MatcherRelated, AttackMixin):
    """Entity matcher for related items that have been trained on."""

    def from_attack(self, attack: ATTACK, verbose: bool=False) -> 'AttackMixin':
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

        # Group sentences per label
        sentences = defaultdict(list)
        for sentence, label in attack2phrases(attack):
            sentences[label].append(sentence)

        # Create iterator
        iterator = sentences.items()
        # Add verbosity if necessary
        if verbose: iterator = tqdm(
            iterator,
            desc = "Loading MatcherMitreRelated"
        )

        # Add all sentences and labels
        for match_id, sentences_ in iterator:
            self.add_sentences(
                match_id=match_id,
                sentences = sentences_
            )

        # Return self
        return self