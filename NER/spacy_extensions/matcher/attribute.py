from collections import defaultdict
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
from typing import Any, Dict, List, Tuple, Union


class AttributeMatcher:
    """Fast SpaCy matcher based on a single token attribute."""

    def __init__(
            self,
            vocab: Vocab,
            attribute: str,
            validate: bool = True,
        ):
        """Create the rule-based Matcher. If validate=True is set, all patterns
            added to the matcher will be validated against a JSON schema and a
            MatchPatternError is raised if problems are found. Those can
            include incorrect types (e.g. a string where an integer is
            expected) or unexpected property names.
            
            Parameters
            ----------
            vocab : Vocab
                The vocabulary object, which must be shared with the documents
                the matcher will operate on.

            attribute : str
                Attribute on which to match. For custom attributes use
                `_.attribute`.

            validate : bool
                Validate all patterns added to this matcher.
            """
        self.vocab = vocab
        self.attribute = attribute
        if attribute.startswith('_.'):
            self.attribute_func = lambda x: getattr(x._, attribute[2:])
        else:
            # Cast from tokens such as ORTH, NORM, LOWER, POS, TAG, etc.
            attribute = {a : f"{a.lower()}_" for a in [
                "ORTH", "NORM", "LOWER", "POS", "TAG", "DEP", "LEMMA", "SHAPE",
                "ENT_TYPE", "ENT_IOB", "ENT_ID", "ENT_KB_ID"
            ]}.get(self.attribute, self.attribute)
            self.attribute_func = lambda x: getattr(x, attribute)
        self.validate = validate
        self._patterns = defaultdict(list)
        self.attribute_map = defaultdict(set)

    def __call__(
            self,
            doclike: Union[Doc, Span],
            as_spans: bool = False,
        ) -> Union[List[Tuple[int, int, int]], List[Span]]:
        """Find all token sequences matching the supplied patterns on the Doc
            or Span.

            Note that if a single label has multiple patterns associated with
            it, the returned matches don’t provide a way to tell which pattern
            was responsible for the match.

            Parameters
            ----------
            doclike : Union[Doc, Span]
                The `Doc` or `Span` to match over. 

            as_spans : bool, default=False
                Instead of tuples, return a list of `Span` objects of the
                matches, with the `match_id` assigned as the span label.
                Defaults to `False`.

            Returns
            -------
            result : Union[List[Tuple[int, int, int]], List[Span]]
                A list of `(match_id, start, end)` tuples, describing the
                matches. A match tuple describes a span `doc[start:end]`. The
                match_id is the ID of the added match pattern. If `as_spans` is
                set to `True`, a list of `Span` objects is returned instead. 
            """
        # Get attributes of each token in document
        attributes = map(self.attribute_func, doclike)
        # Find matches
        matches = map(self.attribute_map.get, attributes)
        # Zip token with matches
        matches = zip(doclike, matches)
        # Filter non-matches
        matches = filter(lambda x: x[1], matches)

        # Initialise result
        result = list()
        # Extract matches
        for token, ids in matches:
            for match_id in sorted(ids):
                result.append((match_id, token.i, token.i+1))

        # Convert to spans if required
        if as_spans:
            result = [Span(
                doc = doclike.doc if isinstance(doclike, Span) else doclike,
                start = start,
                end = end,
                label = match_id,
            ) for match_id, start, end in result]

        # Return result
        return result

    def __len__(self) -> int:
        """Get the number of rules added to the matcher. Note that this only
            returns the number of rules (identical with the number of IDs), not
            the number of individual patterns.

            Returns
            -------
            The number of rules.
        """
        return len(self._patterns)

    def __contains__(self, key: str) -> bool:
        """Check whether the matcher contains rules for a match ID.
        
            Parameters
            ----------
            key : str
                The match ID.
                
            Returns
            -------
            result : bool
                Whether the matcher contains rules for this match ID.
            """
        return key in self._patterns

    def add(
            self,
            match_id: str,
            patterns: List[List[Dict[str, Any]]],
            on_match: Any = None,
            greedy: Any = None,
        ) -> None:
        """Add a rule to the matcher, consisting of an ID key, one or more
            patterns, and an optional callback function to act on the matches.
            The callback function will receive the arguments matcher, doc, i
            and matches. If a pattern already exists for the given ID, the
            patterns will be extended. An on_match callback will be overwritten.

            Parameters
            ----------
            match_id : str
                An ID for the thing you're matching. 

            patterns : List[List[Dict[str, Any]]]
                Match pattern. A pattern consists of a list of dicts, where each
                dict describes a token. Note that patterns must be compatible 
                with attribute set in Matcher. If pattern is not compatible, 
                this method will throw a ValueError.

            on_match : Any
                Ignored, only here for Matcher API compatibility.

            greedy : Any
                Ignored, only here for Matcher API compatibility.
            """
        # Add patterns
        self._patterns[match_id].extend(patterns)

        # Get value of match_id
        match_id = self.vocab[match_id].orth

        # Loop over all patterns for given key
        for pattern in patterns:
            # Loop over token_patterns
            for token_pattern in pattern:
                # Assert token pattern matches self.attribute
                if len(token_pattern) != 1:
                    raise ValueError(
                        f"Multiple token conditions not supported: "
                        f"'{token_pattern}'"
                    )
                
                # Custom attribute
                if self.attribute.startswith('_.'):
                    if ('_' not in token_pattern or
                        self.attribute[2:] not in token_pattern['_']):
                        raise ValueError(
                            f"Attribute '{self.attribute}' not present"
                        )
                    token_pattern = token_pattern['_']
                    if len(token_pattern) != 1:
                        raise ValueError(
                            f"Multiple token conditions not supported: "
                            f"'{token_pattern}'"
                        )
                    token_pattern = token_pattern[self.attribute[2:]]

                # Get direct attribute
                else:
                    if self.attribute not in token_pattern:
                        raise ValueError(
                            f"Attribute '{self.attribute}' not present"
                        )
                    token_pattern = token_pattern[self.attribute]
                    
                # Case of value
                if isinstance(token_pattern, str):
                    token_patterns = [token_pattern]
                elif isinstance(token_pattern, dict):
                    if len(token_pattern) != 1:
                        raise ValueError(
                            f"Multiple token conditions not supported: "
                            f"'{token_pattern}'"
                        )
                    if 'IN' not in token_pattern:
                        raise ValueError(
                            f"Unsupported pattern '{token_pattern}'. Must be "
                            "Dict[Literal['IN'], List[str]]."
                        )
                    token_patterns = token_pattern['IN']
                else:
                    raise ValueError(
                        f"Unsupported pattern '{token_pattern}'. Must be "
                        "either of type str or Dict[Literal['IN'], List[str]]."
                    )
                
                for token_pattern in token_patterns:
                    self.attribute_map[token_pattern].add(match_id)

    def remove(self, key: str) -> None:
        """Remove a rule from the matcher. A `KeyError` is raised if the match
            ID does not exist.
            
            Parameters
            ----------
            key : str
                The ID of the match rule.
            """
        del self._patterns[key]
        for attribute, mapping in self.attribute_map:
            if key in mapping:
                self.attribute_map[attribute] = {x for x in mapping if x != key}

    def get(self, key: str) -> List[List[Dict[str, Any]]]:
        """Retrieve the pattern stored for a key. Returns the rule as an
            `(on_match, patterns)` tuple containing the callback and available
            patterns.

            Parameters
            ----------
            key : str
                The ID of the match rule. 
            
            Returns
            -------
            result : List[List[Dict[str, Any]]]
                Patterns for given key.
            """
        return self._patterns[key]
