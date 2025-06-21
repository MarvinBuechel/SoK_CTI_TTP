# Imports
import itertools
import spacy
from collections import Counter
from spacy.tokens import Doc, Span, Token
from spacy_extensions.utils import check_dependencies
from spacy_extensions.utils import SpacyCallable
from typing import Optional

@spacy.Language.factory('matcher_coref')
class MatcherCoref(SpacyCallable):
    """Add entity references to tokens matched via coreference resolution.
        Coreference occurs when two or more expressions refer to the same person
        or thing. Coreference resolution aims to connect tokens pointing to the
        same person or thing.

        The MatcherCoref uses the out-of-the-box coreference resolution library
        `coreferee <https://spacy.io/universe/project/coreferee>`_ to find
        tokens pointing to the same object, and attaches entity labels to tokens
        if any of the tokens contains an entity label.
        
        Dependencies
        ------------
        coreferee
            https://spacy.io/universe/project/coreferee

        Extensions
        ----------
        coref_chain : Token
            Retrieves the coreference chain for the given token if any. See
            :py:meth:`coref_chain`.

        coref_root : Token
            Retrieves the root token that a token is refering to if the token is
            in any coref_chain. See :py:meth:`coref_root`.
        """

    def __init__(self, nlp, name, force: bool = True):
        """Create new MatcherCoref object.
        
            Parameters
            ----------
            nlp : spacy.Language
                Language pipeline in which MatcherRegex is used.

            name : string
                Name of pipeline.

            force : bool, default=True
                If True, overwrite existing extensions.
            """
        # Check dependencies
        check_dependencies(
            nlp  = nlp,
            name = name,
            doc  = [('coref_chains', 'coreferee')],
        )

        # Set extensions
        if force or not Token.has_extension('coref_chain'):
            Token.set_extension(
                name   = 'coref_chain',
                getter = self.coref_chain,
                force = force,
            )

        if force or not Token.has_extension('coref_root'):
            Token.set_extension(
                name   = 'coref_root',
                getter = self.coref_root,
                force = force,
            )

    ########################################################################
    #                                 Call                                 #
    ########################################################################

    def __call__(self, doc: Doc) -> Doc:
        """Add entities based on coreference chains.
        
            Parameters
            ----------
            doc : Doc
                Document in which to add entities based on coreference chains.
                
            Returns
            -------
            doc : Doc
                Document where entities are added based on coreference chains.
            """
        # Initialise spans
        spans = list()

        # Loop over each coreference chain
        for chain in doc._.coref_chains:
            # Extract tokens from chain
            indices = map(lambda mention: mention.token_indexes, chain)

            # Get corresponding tokens in document
            tokens = map(lambda index: doc[index], itertools.chain(*indices))
            tokens = list(tokens)
            
            # Get most specific mention index
            target = doc[chain[chain.most_specific_mention_index].token_indexes[0]]

            # Get target ent type
            target_ent = target.ent_type_

            # If the target does not have an ent type,
            # get the most common ent type
            if not target_ent:
                entities = map(lambda token: token.ent_type_, tokens)
                target_ent = Counter(entities).most_common(1)[0][0]

            if target_ent:
                for token in tokens:
                    if not token.ent_type_:
                        spans.append(Span(
                            doc   = doc,
                            start = token.i,
                            end   = token.i + 1,
                            label = target_ent,
                        ))

        # Set document entities
        doc.set_ents(
            spacy.util.filter_spans(
                tuple(filter(None, spans)) +
                doc.ents
            )
        )

        # Return document
        return doc

    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def coref_chain(self, token: Token) -> Optional[object]:
        """Get coref chain corresponding to token.
        
            Parameters
            ----------
            token : Token
                Token for which to retrieve coref_chain.

            Returns
            -------
            chain : Optional[Chain]
                Relevant coreference chain if any.
            """
        # Loop over all coref chains in document
        for chain in token.doc._.coref_chains:
            # Get indices of tokens in chain
            chain_indices = itertools.chain(*map(
                lambda mention: mention.token_indexes,
                chain,
            ))

            # Check if token is in chain
            if token.i in chain_indices:
                # Return relevant chain
                return chain


    def coref_root(self, token: Token) -> Optional[Token]:
        """Get root that given token is referencing to, if any.
            E.g., in the sentence ``Jack is tired, he goes to bed.`` the
            ``coref_root`` of ``he`` is ``Jack``.
        
            Parameters
            ----------
            token : Token
                Token for which to retrieve coref_root.

            Returns
            -------
            root : Optional[Token]
                Coref root, if any.
            """
        # Get relevant chain
        chain = token._.coref_chain

        # Return root if any
        if chain:
            print(chain)
            root = chain[chain.most_specific_mention_index].token_indexes[0]
            return token.doc[root]
