# Imports
from functools import partial
import base64
import json
from pathlib   import Path
from py_attack import ATTACK
from typing    import Callable, Iterable, Optional, Union
import re
import spacy
from spacy.tokens import Doc, Token, Span
from spacy.pipeline.entityruler import DEFAULT_ENT_ID_SEP, entity_ruler_score
from spacy_extensions.pipeline.matchers.phrase import MatcherPhrases
from spacy_extensions.utils.matcher import AttackMixin

################################################################################
#                                     Pipe                                     #
################################################################################

@spacy.Language.factory('matcher_mitre_attack')
class MatcherMitreAttack(MatcherPhrases, AttackMixin):
    """Entity matcher for MITRE ATT&Ck concepts via MatcherPhrases.
        This class adds a detector for MITRE ATT&CK concepts based on the names
        and aliases defined in the MITRE ATT&CK framework.
        
        Superclasses
        ------------
        MatcherPhrases
            The MatcherMitreAttack is a subclass of :py:class:`MatcherPhrases`.

        Extensions
        ----------
        mitre_attack : Doc
            Returns an iterator over all MITRE ATTACK entities found in
            document. See :py:meth:`filter_ents`.
        
        attack_type : Union[Span, Token]
            Returns the MITRE ATTACK label corresponding to the Span or Token
            entity, if any. See :py:meth:`attack_type`.
        
        attack_concept : Union[Span, Token]
            Returns the MITRE ATTACK concept corresponding to the Span or Token
            entity, if any. See :py:meth:`attack_concept`.
        """
    
    def __init__(self,
            nlp,
            name,
            phrase_matcher_attr: Optional[Union[int, str]] = None,
            validate           : bool = False,
            overwrite_ents     : bool = False,
            ent_id_sep         : str = DEFAULT_ENT_ID_SEP,
            scorer             : Optional[Callable] = None,
        ):
        """Create an entity ruler for the MITRE ATT&CK concepts."""
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

        # Set ATT&CK concept regular expression
        base_regex = r'((TA|T|S|G|M|DS)\d{4}(\.\d{3})?)'
        self.attack_regex = re.compile(
             rf'^{base_regex}({re.escape(self.ent_id_sep)}{base_regex})*$'
        )

        # Register additional document extensions
        Doc.set_extension(
            name   = 'mitre_attack',
            getter = partial(
                self.filter_ents,
                regex = self.attack_regex,
            )
        )

        # Return ATT&CK type if any
        Span.set_extension(
            name = 'attack_type',
            getter = self.attack_type,
        )
        Token.set_extension(
            name = 'attack_type',
            getter = self.attack_type,
        )
        Span.set_extension(
            name = 'attack_concept',
            getter = self.attack_concept,
        )
        Token.set_extension(
            name = 'attack_concept',
            getter = self.attack_concept,
        )

    ########################################################################
    #                              Extensions                              #
    ########################################################################

    def filter_ents(self, doc: Doc, regex: re.Pattern) -> Iterable[Span]:
        """Filter entities based on given regex pattern.
        
            Parameters
            ----------
            doc : Doc
                Document in which to filter the entities.
            
            regex : re.Pattern
                Pattern on which to filter entities.
            """
        yield from filter(
            lambda x: regex.match(self.nlp.vocab[x.label].text),
            doc.ents,
        )


    def attack_type(self, obj: Union[Span, Token]) -> str:
        """Get the MITRE ATTACK type from a given span or token.
        
            Parameters
            ----------
            obj : Union[Span, Token]
                SpaCy obj for which to retrieve ATTACK type.
            
            Returns
            -------
            type : str
                Name of ATTACK type.
            """
        # Retrieve IOC type
        if isinstance(obj, Span):
            type = self.nlp.vocab[obj.label].text
        else:
            type = obj.ent_type_

        # Check if IOC type is of valid type and return
        return type if isinstance(type, str) and self.attack_regex.match(type) else None


    def attack_concept(self, obj: Union[Span, Token]) -> Optional[dict]:
        """Get the MITRE ATTACK concept from a given span or token.
        
            Parameters
            ----------
            obj : Union[Span, Token]
                SpaCy obj for which to retrieve ATTACK concept.
            
            Returns
            -------
            concept : Optional[dict]
                Name of ATTACK type.
            """
        # Get ATT&CK type
        attack_type = obj._.attack_type

        # In case we found an ATT&CK type
        if attack_type is not None:
            # Return Dictionary of ATT&CK concepts
            return {
                concept: self.attack.get(concept)
                for concept in self.attack_regex.match(attack_type).groups()
                if self.attack.get(concept)
            }
        
        # Return None by default
        return None

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_bytes(self, *, exclude: Iterable[str] = ...) -> bytes:
        """Store MatcherMitreAttack as bytes.
        
            Note
            ----
            While this method works, we recommend to use to_disk whenever
            possible as it produces much smaller files. This method is merely
            added for compatibility.
            """
        # In case we should include self.attack
        if exclude is ... or (exclude and 'attack' not in exclude):
            # Get result from super
            entity_ruler = super().to_bytes(exclude=exclude)

            # Return as bytes
            return json.dumps({
                'entity_ruler': base64.b64encode(entity_ruler).decode('utf-8'),
                'attack'      : self.attack.to_json(),
            }).encode('utf-8')

        # Otherwise, call EntityRuler.to_bytes
        else:
            return super().to_bytes(exclude=exclude)


    def from_bytes(self, patterns_bytes: bytes, *, exclude: Iterable[str] = ...):
        """Load MatcherMitreAttack from bytes.
        
            Note
            ----
            While this method works, we recommend to use from_disk whenever
            possible as it deals with much smaller files. This method is merely
            added for compatibility.
            """
        # In case we should include self.attack
        if exclude is ... or (exclude and 'attack' not in exclude):
            # Load data
            data = json.loads(patterns_bytes.decode('utf-8'))

            # Load attack
            self.attack = ATTACK.from_json(data['attack'])

            # Load super
            return super().from_bytes(
                base64.b64decode(data['entity_ruler'].encode('utf-8')),
                exclude = exclude,
            )

        # Otherwise, call EntityRuler.from_bytes
        else:
            return super().from_bytes(exclude=exclude)


    def to_disk(self, path: Union[str, Path], *, exclude: Iterable[str] = ...) -> None:
        """Store MatcherMitreAttack as file on disk.
            Equivalent to EntityRuler.to_disk, with the addition that it stores
            the self.attack variable as well.
            """
        # Store variables as in super
        result = super().to_disk(path, exclude=exclude)

        # Cast path to Path            
        path = spacy.util.ensure_path(path)

        # Store self.attack if necessary
        if exclude and 'attack' not in exclude:
            with open(path / 'attack.json', 'w') as outfile:
                outfile.write(self.attack.to_json())

        # Return result
        return result


    def from_disk(self, path: Union[str, Path], *, exclude: Iterable[str] = ...):
        """Load MatcherMitreAttack from file on disk.
            Equivalent to EntityRuler.from_disk, with the addition that it loads
            the self.attack variable as well.
            """
        # Load variables as in super
        result = super().from_disk(path, exclude=exclude)

        # Cast path to Path            
        path = spacy.util.ensure_path(path)

        # Load self.attack if necessary
        if exclude and 'attack' not in exclude:
            with open(path / 'attack.json') as infile:
                self.attack = ATTACK.from_json(infile.read())

        # Return result
        return result
