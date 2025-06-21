# Imports
import json
import pandas as pd
import spacy
from spacy.matcher import DependencyMatcher
from spacy.tokens  import Doc, Token, Span
from spacy_extensions.utils import SpacyCallable, SpacySerializable
from tqdm import tqdm
from typing import Iterable, List, Set, Tuple

################################################################################
#                                     Pipe                                     #
################################################################################

@spacy.Language.factory('matcher_ttpdrill')
class MatcherTTPDrill(SpacyCallable, SpacySerializable):
    """Entity matcher for TTPDrill concepts via MatcherPhrases.
        Based on the paper "TTPDrill: Automatic and Accurate Extraction of
        Threat Actions from Unstructured Text of CTI Sources."
        DOI: https://doi.org/10.1145/3134600.3134646

        This matcher uses the TTPDrill ontology as defined in the 
        https://github.com/KaiLiu-Leo/TTPDrill-0.5 repository.
        
        Superclasses
        ------------
        DependencyMatcher
            The MatcherTTPDrill is a subclass of SpaCy's
            `DependencyMatcher <https://spacy.io/api/dependencymatcher>`_.

        Requires
        --------
        matcher_dict : Token
            Matcher dict for matching token, provided by spacy_extensions.
            pipeline.TokenBase.

        Extensions
        ----------
        ttpdrill : Doc
            Returns an iterator over all TTPDrill entities found in
            document. See :py:meth:`get_matches`.

        ttpdrill_raw : Doc
            Returns an iterator over all TTPDrill entities, returned as raw
            matcher results. See :py:meth:`get_matches_raw`.

        ttpdrill_sents : Doc
            Returns all sentences where TTPDrill detected a MITRE ATT&CK
            technique, including the labels it detected.
            See :py:meth:`get_matches_sents`.
        """
    
    def __init__(self, nlp, name):
        """Create an entity ruler for the TTPDrill concepts."""
        # Set nlp and name
        self.nlp  = nlp
        self.name = name

        # Initialise matcher
        self.matcher  = DependencyMatcher(nlp.vocab, validate=True)
        self.patterns = list()

        # Register document extensions
        Doc.set_extension(
            name   = 'ttpdrill',
            getter = self.get_matches,
        )

        Doc.set_extension(
            name   = 'ttpdrill_raw',
            getter = self.get_matches_raw,
        )

        Doc.set_extension(
            name   = 'ttpdrill_sents',
            getter = self.get_matches_sent,
        )

    ########################################################################
    #                                 Call                                 #
    ########################################################################

    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def get_matches_raw(self, doc: Doc) -> List[Tuple[int, List[int]]]:
        """Return matches for given document.
        
            Parameters
            ----------
            doc : Doc
                Document for which to find matches.
                
            Returns
            -------
            matches : List[Tuple[int, List[int]]]
                See ``DependencyMatcher.__call__``
            """
        return self.matcher(doc)

    def get_matches(self, doc: Doc) -> Iterable[Tuple[str, List[Token]]]:
        """Return TTPDrill matches for given document.
            
            Parameters
            ----------
            doc : Doc
                Document for which to find matches.
                
            Yields
            ------
            label : str
                Label corresponding to match

            match : List[Token]
                Tokens in doc that matched.
            """
        # Iterate over matches
        for label, match in doc._.ttpdrill_raw:
            # Get label
            label = self.nlp.vocab[label].text
            # Get match
            match = [doc[index] for index in match]

            # Yield match
            yield label, match

    def get_matches_sent(self, doc: Doc) -> Iterable[Tuple[Set[str], Span]]:
        """Return TTPDrill matches for given document.
            
            Parameters
            ----------
            doc : Doc
                Document for which to find matches.
                
            Yields
            ------
            label : Set[str]
                Labels corresponding to sentence

            span : Span
               Sentence in doc that matched.
            """
        # Initialise result
        result = dict()

        # Iterate over matches
        for label, match in doc._.ttpdrill:
            # Get sentence
            sentence = match[0].sent

            # Get key corresponding to sentence
            key = sentence[0].i

            # Add sentence if it does not yet exist
            if key not in result:
                result[key] = {
                    'labels': set(),
                    'sentence': sentence,
                }

            # Append label
            result[key]['labels'].add(label)

        # Yield matches
        for _, result in sorted(result.items()):
            yield result['labels'], result['sentence']

    ########################################################################
    #                           Register methods                           #
    ########################################################################

    def from_ontology(self, ontology: pd.DataFrame, verbose: bool=False) -> None:
        """Load phrases from given MITRE ATT&CK framework.
        
            Parameters
            ----------
            ontology : ATTACK
                Attack framework from which to load phrases.
        """
        # Get iterator
        iterator = ontology.iterrows()

        # Add verbosity, if necessary
        if verbose:
            iterator = tqdm(
                iterator,
                desc  = 'Parsing ontology',
                total = ontology.shape[0],
            )

        # Iterate over all rows
        for _, row in iterator:
            pattern = [
                row['id'].upper(),    # Label
                [self.action2pattern( # Pattern
                    action = str(row['what']),
                    target = str(row['where']),
                )],
            ]

            # Add pattern
            self.matcher.add(pattern[0], pattern[1])
            self.patterns.append(pattern)

    ########################################################################
    #                          Auxiliary methods                           #
    ########################################################################

    def action2pattern(self, action: str, target: str) -> List[dict]:
        """Transform an (action, target) tuple to a phrase match.
            The main idea is that the action is the head in a chain to the
            target following head -> dependency path: action >> target.
        
            Parameters
            ----------
            action : str
                Action (verb) from which to generate a matcher pattern.
            
            target : str
                Target (noun) from which to generate a matcher pattern.
            
            Returns
            -------
            pattern : List[dict]
                Resulting matcher pattern.
            """
        # Initialise pattern
        pattern = list()

        # Disable pipe, if required
        enabled = self.name in dict(self.nlp.pipeline)
        if enabled: self.nlp.disable_pipe(self.name)

        # Create pattern
        action = self.nlp(action)
        target = self.nlp(target)

        # Add action
        for index, token in enumerate(action):

            # Set root
            if index == 0:
                pattern.append({
                    'RIGHT_ID': 'verb',
                    'RIGHT_ATTRS': token._.matcher_dict,
                })

            # Otherwise add additional requirements
            else:
                pattern.append({
                    'LEFT_ID': 'verb',
                    'REL_OP' : '>>',
                    'RIGHT_ID': f"ACTION_{token.i}",
                    'RIGHT_ATTRS': token._.matcher_dict,
                })

        # Add targets
        for token in target:
            pattern.append({
                'LEFT_ID': 'verb',
                'REL_OP' : '>>',
                'RIGHT_ID': f"TARGET_{token.i}",
                    'RIGHT_ATTRS': token._.matcher_dict,
            })

        # Re-enable pipe
        if enabled: self.nlp.enable_pipe(self.name)

        # Return pattern
        return pattern

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_bytes(self, *args, **kwargs) -> bytes:
        """Represent trained TTPDrill object as bytes.
        
            Parameters
            ----------
            *args: Ignored
            
            **kwargs: Ignored
            
            Returns
            -------
            result : bytes
                Bytes representation of TTPDrill.
            """
        # Convert to bytes using json
        return json.dumps(self.patterns).encode('utf-8')

    def from_bytes(self, data: bytes, *args, **kwargs):
        """Load trained TTPDrill object from bytes.
        
            Parameters
            ----------
            data : bytes
                Bytes from which to load TTPDrill.

            *args: Ignored
            
            **kwargs: Ignored
            """
        # Load patterns
        patterns = json.loads(data.decode('utf-8'))

        # Add patterns
        for label, pattern in patterns:
            self.matcher.add(label, pattern)

if __name__ == "__main__":
    from pathlib import Path
    from glob import glob
    nlp = spacy.load('en_core_web_sm')
    ttpdrill = nlp.add_pipe('matcher_ttpdrill')

    # Load ontology from file
    base = Path('/home/thijs/Documents/research/eagle/tools/TTPDrill-0.5/ontology')
    # verbs = pd.read_csv(base / 'verb_list.txt', names=['what'])
    how   = pd.read_csv(base / 'refined_how.csv')
    what  = pd.read_csv(base / 'refined_what_where.csv')
    why   = pd.read_csv(base / 'refined_why.csv')

    # Create ontology based on how, what and why
    ontology = pd.concat((how, what, why), ignore_index=True)
    
    # Create TTPDrill instance from ontology
    ttpdrill.from_ontology(ontology, verbose=True)
    ttpdrill.to_disk('temp')
    ttpdrill.from_disk('temp')

    for filename in glob('/home/thijs/Documents/research/eagle/data/mitre-dataset/text/e48*'):
        with open(filename) as infile:
            text = infile.read()

        doc = nlp(text)
        for label, match in doc._.ttpdrill_sents:
            print(label, match)
        print()