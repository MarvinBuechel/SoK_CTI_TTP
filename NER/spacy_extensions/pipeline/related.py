import spacy
from nltk.corpus import wordnet as wn
from spacy.tokens import Token
from spacy_extensions.utils.callable import SpacyCallable
from spacy_extensions.utils.wordnet import POS_SPACY_TO_WN
from typing import Set

@spacy.Language.factory('related',
    assigns  = ['token._.related', 'token._.synsets'],
    requires = ['token._.pos', 'token._.token_base'],
)
class RelatedTokens(SpacyCallable):
    """Identifies related tokens for each token.

        Extensions
        ----------
        related : Token
            Set of related tokens by text.
        """

    def __init__(self, nlp, name, force: bool = True):
        """RelatedTokens constructor.

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

        # Set base extension
        if force or not Token.has_extension('related'):
            Token.set_extension(
                name   = 'related',
                getter = self.related_wn,
                force = force,
            )

        if force or not Token.has_extension('synsets'):
            Token.set_extension(
                name = 'synsets',
                getter = lambda token: set(wn.synsets(
                    lemma = token.text,
                    # pos = POS_SPACY_TO_WN.get(token._.pos or token.pos_),
                )),
                force = force,
            )

    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def related_wn(self, token: Token) -> Set[str]:
        """Find wordnet tokens that are related to the actual token, using the 
            wordnet dictionary.
            
            Parameters
            ----------
            token : Token
                Token for which to find related wordnet token.

            Returns
            -------
            related : Set[str]
                String representation of each related token.
            """
        # Get set of related tokens
        related = set([token._.token_base])

        # Ignore related proper nouns
        if (token._.pos or token.pos_) != 'PROPN':
            # Loop over all representations of token
            for text in [token.text, token._.token_base]:
                # Get synonyms of token
                for synset in wn.synsets(text,
                        pos = POS_SPACY_TO_WN.get(token._.pos or token.pos_)
                    ):
                    for lemma in synset.lemmas():
                        related.add(lemma.name())
                        related.add(wn.morphy(lemma.name()))
        
        # Return related tokens
        return {term for term in related if term and '_' not in term}
