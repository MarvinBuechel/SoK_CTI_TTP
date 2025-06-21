from typing import Any, Dict
import spacy
from nltk.corpus import wordnet as wn
from spacy.tokens import Token
from spacy_extensions.utils.callable import SpacyCallable

@spacy.Language.factory('token_base',
    assigns  = ['Token._.token_base', 'Token._.matcher_dict'],
    requires = ['Token._.pos', 'Token._.related'],
)
class TokenBase(SpacyCallable):
    """Set the Token._.matcher_dict attribute according to given configuration.
        Used by other Matchers (e.g., Phrase and Fuzzy matchers) to find the
        base value of the token against which to match.

        Extensions
        ----------
        token_base : Token
            Base form of token to use for matching.

        matcher_dict : Token
            Base form of token to use for matching.
        """

    def __init__(self, nlp, name, force: bool = True):
        """MatcherExact constructor.

            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which MatcherExact is used.

            name : string
                Name of pipeline.
            """
        # Set pipeline
        self.nlp  = nlp
        self.name = name

        # Set base extension
        if force or not Token.has_extension('token_base'):
            Token.set_extension(
                name   = 'token_base',
                getter = self.base_token,
                force = force,
            )

        # Set token matcher extension
        if force or not Token.has_extension('matcher_dict'):
            Token.set_extension(
                name   = 'matcher_dict',
                getter = self.base_dict,
                force = force,
            )

        # Set editable POS extension
        if force or not Token.has_extension('pos'):
            Token.set_extension('pos', default=None, force=force)

    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def base_token(self, token : Token) -> str:
        """Create a base for a specific token.
        
            Parameters
            ----------
            token : Token
                Token for which to create a base.
                
            Returns
            -------
            base : str
                Base form of token.
            """
        return (
            wn.morphy(token.lemma_ or token.norm_) or
                     (token.lemma_ or token.norm_).lower()
        )

    def base_dict(self, token : Token) -> Dict[str, Any]:
        """Return a dictionary object that can be used as a SpaCy matcher
            RIGHT_ATTRS value. The full format that can be returned is 
            specified at https://spacy.io/api/matcher
            
            Parameters
            ----------
            token : Token
                Token for which to create a base matcher dictionary.
            
            Returns
            -------
            base_dict : Dict[str, Any]
                Dict for SpaCy matcher RIGHT_ATTRS value. The full format that
                can be returned is specified at https://spacy.io/api/matcher
            """
        # Initialise result
        result = {
            # '_': {'token_base': token._.token_base}
            '_': {'token_base': {"IN": list(token._.related)}}
        }

        # Add POS check in case of custom set PROPN
        if token._.pos == 'PROPN':
            result['POS'] = (token._.pos or token.pos_)

        # Return result
        return result