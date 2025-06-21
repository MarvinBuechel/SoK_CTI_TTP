from collections import defaultdict
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union
)

class AttributeMatcher:
    """A faster Matcher implementation.
        Only supports a single attribute for a single token, but is much
        quicker than the regular spacy.matcher.Matcher.
        """

    def __init__(
            self,
            vocab: Vocab,
            attribute: str,
            custom  : bool = False,
            validate: bool = True,
            relative: bool = False,
        ):
        """Initialise AttributeMatcher.
            
            Parameters
            ----------
            vocab : Vocab
                The vocabulary object, which must be shared with the documents
                the matcher will operate on.
            
            attribute : str
                The attribute on which to match.
                
            custom : bool, default=False
                Set to True if given attribute should be considered a custom
                attribute.
            
            validate : bool, default=True
                Validate all patterns added to this matcher.

            relative : bool, default=False
                If True, returns matches relative to given doclike start. 
                Otherwise, matches are returned as indices of outer doc.
            """
        # Initialise matcher attributes
        self.vocab = vocab
        self.validate = validate
        self.custom = custom
        self.attribute = attribute
        self.relative = relative

        # Initialise patterns
        self.patterns = defaultdict(set)
        # Initialise callbacks
        self.callbacks = dict()

    @property
    def match_ids(self) -> Set[str]:
        """all match IDs on which the matcher was trained."""
        return set.union(*self.patterns.values())

    def __call__(
            self,
            doclike: Union[Doc, Span],
            as_spans: bool = False,
            allow_missing: bool = False,
            with_alignments: bool = False,
        ) -> Union[List[Tuple[int, int, int]], List[Span]]:
        """Find all token sequences matching the supplied patterns on the Doc
            or Span.

            Note that if a single label has multiple patterns associated with
            it, the returned matches don’t provide a way to tell which pattern
            was responsible for the match.
            
            Parameters
            ----------
            doclike : Union[Doc, Span]
                The Doc or Span to match over.
            
            as_spans : bool, default=False
                Instead of tuples, return a list of Span objects of the
                matches, with the match_id assigned as the span label. Defaults
                to False.

            allow_missing : bool, default=False
                Whether to skip checks for missing annotation for attributes
                included in patterns. Defaults to False.

            with_alignments : bool, default=False
                Currently ignored.
                Return match alignment information as part of the match tuple
                as List[int] with the same length as the matched span. Each
                entry denotes the corresponding index of the token in the
                pattern. If as_spans is set to True, this setting is ignored.
                Defaults to False. 
            """
        # Initialise result
        matches = list()

        # Loop over tokens in doc
        for token in doclike:
            # Initialise attribute
            attribute = None

            # Get attribute of token
            if self.custom:
                if hasattr(token._, self.attribute):
                    attribute = getattr(token._, self.attribute)
            else:
                if hasattr(token, self.attribute):
                    attribute = getattr(token, self.attribute)

            # Perform check if not allow_missing
            if not allow_missing and attribute is None:
                raise ValueError(
                    f"Token does not contain {'custom ' if self.custom else ''}"
                    f"attribute '{self.attribute}'."
                )

            # Get corresponding matching patterns
            for match_id in self.patterns.get(attribute, list()):
                # Add relative match to result (relative to start of doclike)
                if self.relative:
                    matches.append((
                        self.vocab[match_id].orth,
                        token.i - doclike[0].i,
                        token.i - doclike[0].i + 1,
                    ))
                # Add absolute match to result (absolute w.r.t. doclike)
                else:
                    matches.append((
                        self.vocab[match_id].orth,
                        token.i,
                        token.i + 1,
                    ))


        # Transform matches to spans
        if as_spans:
            # Assert matches is not relative
            if self.relative:
                raise ValueError(
                    "Cannot match as_spans when self.relative = True."
                )

            # In case doclike is a Span, get the corresponding doc
            if isinstance(doclike, Span):
                doclike = doclike.doc

            # Transform matches to Span objects
            matches = [
                Span(doclike, start, end, label)
                for label, start, end in matches
            ]

        # Return result
        return matches

    def __len__(self) -> int:
        """Get number of distinct rules that are trained on."""
        return len(self.match_ids)

    def __contains__(self, key: str) -> bool:
        """Check if patterns contains key."""
        return key in self.match_ids

    def add(
            self,
            match_id: str,
            patterns: List[List[Dict[str, Any]]],
            on_match: Optional[
                Callable[['AttributeMatcher', Doc, int, List[tuple]], Any]
            ] = None,
            greedy: Literal['ERROR', 'FIRST', 'SET'] = 'ERROR',
        ) -> None:
        """Add a rule to the matcher, consisting of an ID key, one or more
            patterns, and an optional callback function to act on the matches.
            The callback function will receive the arguments matcher, doc, i
            and matches. If a pattern already exists for the given ID, the
            patterns will be extended. An on_match callback will be overwritten.
            
            Parameters
            ----------
            match_id : str
                An ID for the thing you’re matching.
                
            patterns : List[List[Dict[str, Any]]]
                Match pattern. A pattern consists of a list of dicts, where
                each dict describes a token.

            on_match: Optional[Callable[['AttributeMatcher', Doc, int, List[tuple]], Any]], default=None
                Callback function to act on matches. Takes the arguments
                matcher, doc, i and matches. 

            greedy: Literal['ERROR', 'FIRST', 'SET'], default='ERROR'
                Filter for greedy matches. `FIRST` returns the first pattern
                added to the matcher, `SET` returns the set of patterns that
                matched, `ERROR` raises a ValueError if a pattern already
                exists.
            """
        # Loop over all given patterns
        for pattern in patterns:
            # Check that only a single token is given
            if self.validate and len(pattern) > 1:
                raise ValueError(
                    f"Pattern contained {len(pattern)} tokens, should. "
                     "A pattern should contain at maximum 1 token."
                )

            # Loop over all tokens in pattern
            for attributes in pattern:
                # Extract attribute if we have a custom attribute
                if self.custom:
                    attributes = attributes.get('_', {})

                # Check if only a single attribute is configured
                if self.validate and len(attributes) > 1:
                    raise ValueError(
                        "Pattern cannot contain multiple attributes"
                    )

                # Check if configured attribute is present
                if self.attribute not in attributes:
                    raise ValueError(
                        f"Pattern should contain attribute '{self.attribute}'."
                    )

                # Add pattern
                self._add_pattern_(
                    match_id=match_id,
                    attribute=attributes[self.attribute],
                    greedy=greedy,
                )

        # Handle on_match
        if on_match is not None:
            # Validate callback
            if self.validate and match_id in self.callbacks:
                raise ValueError(
                    f"There already exists a callback for '{match_id}'"
                )

            # Set callback
            self.callbacks[match_id] = on_match

    def _add_pattern_(
            self,
            match_id: str,
            attribute: Any,
            greedy: Literal['ERROR', 'FIRST', 'SET'],
        ) -> None:
        """Add a single pattern that has been checked and preprocessed to
            matcher.
            
            Note
            ----
            This method should never be called outside of the matcher object. It
            is an internal method that assumes the input has already been
            checked properly. Please always use  the `AttributeMatcher.add()`
            method to add patterns.
            
            Parameters
            ----------
            match_id : str
                An ID for the thing you’re matching.
                
            patterns : Any
                Match pattern. A pattern can be a Dict[str, Any] or a single
                hashable value.

            greedy: Literal['ERROR', 'FIRST', 'SET'], default='ERROR'
                Filter for greedy matches. `FIRST` returns the first pattern
                added to the matcher, `SET` returns the set of patterns that
                matched, `ERROR` raises a ValueError if a pattern already
                exists.
            """
        # Special case for dictionary of properties
        if isinstance(attribute, dict):
            return self._add_pattern_dict_(match_id, attribute, greedy)
        else:
            return self._add_pattern_object_(match_id, attribute, greedy)

    def _add_pattern_dict_(
            self,
            match_id: str,
            attribute: Dict[str, Any],
            greedy: Literal['ERROR', 'FIRST', 'SET'] = 'ERROR',
        ) -> None:
        """Add a single pattern that has been checked and preprocessed to
            matcher.
            
            Note
            ----
            This method should never be called outside of the matcher object. It
            is an internal method that assumes the input has already been
            checked properly. Please always use  the `AttributeMatcher.add()`
            method to add patterns.
            
            Parameters
            ----------
            match_id : str
                An ID for the thing you’re matching.
                
            patterns : Dict[str, Any]
                Match pattern. A pattern can be a Dict[str, Any] or a single
                hashable value.

            greedy: Literal['ERROR', 'FIRST', 'SET'], default='ERROR'
                Filter for greedy matches. `FIRST` returns the first pattern
                added to the matcher, `SET` returns the set of patterns that
                matched, `ERROR` raises a ValueError if a pattern already
                exists.
            """
        # Validate Dict[str, Any] pattern
        if len(attribute) > 1:
            raise ValueError(
                f"Pattern {attribute} should contain only a single value."
            )

        # Process dict
        for key, value in attribute.items():
            if key == 'IN':
                for v in value: self._add_pattern_object_(match_id, v, greedy)
            elif key == 'NOT_IN':
                raise NotImplementedError(f"{key} not yet implemented.")
            elif key == 'IS_SUBSET':
                raise NotImplementedError(f"{key} not yet implemented.")
            elif key == 'IS_SUPERSET':
                raise NotImplementedError(f"{key} not yet implemented.")
            elif key == 'INTERSECTS':
                raise NotImplementedError(f"{key} not yet implemented.")
            elif key == '==':
                raise NotImplementedError(f"{key} not yet implemented.")
            elif key == '>=':
                raise NotImplementedError(f"{key} not yet implemented.")
            elif key == '<=':
                raise NotImplementedError(f"{key} not yet implemented.")
            elif key == '>':
                raise NotImplementedError(f"{key} not yet implemented.")
            elif key == '<':
                raise NotImplementedError(f"{key} not yet implemented.")
            else:
                raise ValueError(f"Unknown dictionary property '{key}")

    def _add_pattern_object_(
            self,
            match_id: str,
            attribute: Any,
            greedy: Literal['ERROR', 'FIRST', 'SET'],
        ) -> None:
        """Add a single pattern that has been checked and preprocessed to
            matcher.
            
            Note
            ----
            This method should never be called outside of the matcher object. It
            is an internal method that assumes the input has already been
            checked properly. Please always use  the `AttributeMatcher.add()`
            method to add patterns.
            
            Parameters
            ----------
            match_id : str
                An ID for the thing you’re matching.
                
            patterns : Any
                Match pattern. A pattern can be a Dict[str, Any] or a single
                hashable value.

            greedy: Literal['ERROR', 'FIRST', 'SET'], default='ERROR'
                Filter for greedy matches. `FIRST` returns the first pattern
                added to the matcher, `SET` returns the set of patterns that
                matched, `ERROR` raises a ValueError if a pattern already
                exists.
            """
        # In case of existing attribute, greedy=ERROR, raise error
        if attribute in self.patterns and greedy == 'ERROR':
            raise ValueError(
                f"Attribute {'_.' if self.custom else ''}"
                f"{self.attribute}={attribute} already exists in matcher."
            )
        # In case of existing attribute, greedy=FIRST, do not update matcher
        elif attribute in self.patterns and greedy == 'FIRST':
            pass
        # Otherwise (non-existing attribute or greedy=SET)
        else:
            self.patterns[attribute].add(match_id)

    def remove(self, key: str) -> None:
        """Remove a rule from the matcher.
            A KeyError is raised if the match ID does not exist.
            
            Parameters
            ----------
            key : str
                The ID of the match rule.
            """
        # Check if matcher contains key
        if key not in self:
            raise KeyError(key)

        # Initialise attribute values to remove
        to_remove = list()
        # Loop over all patterns
        for value, match_ids in self.patterns.items():
            # Check if the pattern matches match_ids
            if key in match_ids:
                # Remove from match
                match_ids.remove(key)
                # Remove value if no match_id is present anymore
                if len(match_ids) == 0:
                    to_remove.append(value)

        # Remove values not containing any more match_id
        for value in to_remove:
            del self.patterns[value]

        # Remove key from 
        if key in self.callbacks:
            del self.callbacks[key]

    def get(self, key: str) -> Tuple[Optional[Callable], List[List[dict]]]:
        """Retrieve the pattern stored for a key. Returns the rule as an
            `(on_match, patterns)` tuple containing the callback and available
            patterns.
            
            Parameters
            ----------
            key : str
                The ID of the match rule. 
            
            Returns
            -------
            rule : Tuple[Optional[Callable], List[List[dict]]]
                The rule, as an `(on_match, patterns)` tuple. 
            """
        # Initialise patterns
        patterns = list()

        # Search all patterns
        for attribute, match_ids in self.patterns.items():
            # Check if key is present
            if key in match_ids:
                # Recreate attribute
                if self.custom:
                    patterns.append([{'_': {self.attribute: attribute}}])
                else:
                    patterns.append([{self.attribute: attribute}])

        # Return result
        return (self.callbacks.get(key), patterns)