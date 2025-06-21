from collections import defaultdict
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from spacy.vocab import Vocab
from spacy_extensions.matcher import AttributeMatcher
from spacy_extensions.utils.tree import min_subtree
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re


class SentenceMatcher:

    def __init__(
            self,
            vocab : Vocab,
            attribute : Optional[str] = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
        """Create base matcher object. All matchers share a vocab.
        
            Parameters
            ----------
            vocab : Vocab
                Vocabulary shared by documents passed to matcher.

            attribute : Optional[str]
                If given, uses the faster AttributeMatcher algorithm for 
                finding matches. The given attribute is used for matching. For
                custom attributes use `_.attribute`. See AttributeMatcher. If
                None is given, uses spacy.matcher.Matcher for matching instead.
            
            *args : Any
                See `spacy.matcher.Matcher`.
            
            **kwargs : Any
                See `spacy.matcher.Matcher`.
            """
        # Initialise Matcher
        self.attribute = attribute
        if attribute is None:
            self._matcher = Matcher(vocab, *args, **kwargs)
        else:
            self._matcher = AttributeMatcher(vocab, attribute, *args, **kwargs)

        # Set checklist
        self.checklist = defaultdict(list)
        # Set patterns
        self.patterns = defaultdict(list)
        # Set pattern regex
        self.regex = re.compile(r'^(.*)_(\d+)_(\d+)$')

    @property
    def vocab(self) -> Vocab:
        """Vocab of underlying matcher. Wrapper for API compatibility."""
        return self._matcher.vocab

    @property
    def patterns_simplified(self) -> Dict[str, List[List[List[str]]]]:
        """Returns a simplified version of patterns.
        
            Returns
            -------
            patterns : Dict[str, List[List[List[str]]]]
                Simplified patterns for each key. Key -> Patterns.
                Each pattern in patterns is in the form `[[X1 or Y1 or ...] AND
                [X2 or Y2 or ...] AND ...]`.
            """
        # Initialise result
        result = defaultdict(list)

        # Loop over all patterns
        for key, value in self.patterns.items():
            # Get all sequences of patterns
            for sequence in value:
                # Simplify sequence
                for index, entry in enumerate(sequence):
                    if '_' not in entry or 'token_base' not in entry['_']:
                        raise ValueError(
                            "simplified pattern only defined for _.token_base"
                        )
                    entry = entry['_']['token_base']
                    entry = entry['IN'] if isinstance(entry, dict) else [entry]
                    sequence[index] = entry
                # Add pattern sequence
                result[key].append(sequence)
        
        # Return result
        return result

    def __call__(
            self,
            doclike: Union[Doc, Span],
            as_spans: bool = False,
        ) -> Union[List[Tuple[int, int, int]], List[Span]]:
        """Find all token sequences matching the supplied pattern on the `Doc` 
            or `Span`.
            
            Parameters
            ----------
            doclike : Union[Doc, Span]
                The `Doc` or `Span` to match over.

            as_spans : bool, default=False
                 Instead of tuples, return a list of `Span` objects of the
                 matches, with the match_id assigned as the span label.
                 Defaults to False. 

            Returns
            -------
            result : Union[List[Tuple[int, int, int]], List[Span]]
                A list of `(match_id, start, end)` tuples, describing the
                matches. A match tuple describes a span `doc[start:end]`. The
                `match_id` is the ID of the added match pattern. If `as_spans`
                is set to `True`, a list of `Span` objects is returned instead. 
        """
        # Initialize result
        result = list()

        # Loop over all sentences
        for sentence in doclike.sents:
            # Get matches
            matches = self._matcher.__call__(sentence, as_spans=True)

            # Continue in case of no matches
            if not matches: continue

            # Process checklist
            checklist = dict()
            for span in matches:
                # Assert match is only a single token
                assert len(span) == 1, "Multitoken matching not supported."

                # Unpack label
                match = self.regex.match(span.label_)
                label, i, j = match.group(1), match.group(2), match.group(3)
                i, j = int(i), int(j)

                # Add match to checklist
                if label not in checklist:
                    checklist[label] = dict()

                if i not in checklist[label]:    
                    checklist[label][i] = [
                        set() for _ in range(self.checklist[label][i])
                    ]
                    
                checklist[label][i][j].add(span.start)

            # Find completed checklists
            for label, patterns in checklist.items():
                for pattern in patterns.values():
                    # Check if we can make at least one match
                    if all(len(token) for token in pattern):
                        # Create match from found pattern
                        created_matches = self._create_matches_(
                            pattern = pattern,
                            label = self.vocab[label].orth,
                            span = sentence,
                        )
                        # Add found matches
                        if len(created_matches) == 1:
                            result.append(created_matches[0])
                        elif len(created_matches) > 1:
                            result.extend(created_matches)

        # Transform to spans if required
        if as_spans:
            result = [Span(
                doc   = doclike.doc,
                start = start,
                end   = end,
                label = label,
            ) for label, start, end in result]

        # Return result
        return result

    def _create_matches_(
            self,
            pattern: List[Set[int]],
            label: int,
            span: Span,
        ) -> List[Tuple[int, int, int]]:
        """Create a match from the given match pattern.
        
            Parameters
            ----------
            pattern : List[Set[int]]
                Matched pattern as found in self.checklist.
                
            label : int
                Label for match.
                
            span : Span
                Span in which match was found.
                
            Returns
            -------
            result : List[Tuple[int, int, int]]
                Found `(match_id, start, end)` tuples.
            """
        # Initialise result
        result = list()

        # Append spans while patterns exist
        while all(len(token) for token in pattern):
            # Find minimal spanning subtree
            span_ = min_subtree(
                sentence = span,
                includes = [[span.doc[i] for i in part] for part in pattern]
            )

            # Stop if we couldn't find a new span
            if span_ is None: return result

            # Append span
            result.append((label, span_.start, span_.end))

            # Remove used patterns
            pattern = [{
                index for index in part
                if index < span_.start or index >= span_.end
            } for part in pattern]

        # Return result
        return result

    def add(
            self,
            match_id: str,
            patterns: List[List[Dict[str, Any]]],
            greedy: Optional[str] = None,
        ) -> None:
        """Add a rule to the matcher, consisting of an ID key, one or more
            patterns, and a callback function to act on the matches. The
            callback function will receive the arguments `matcher`, `doc`, `i`
            and `matches`. If a pattern already exists for the given ID, the
            patterns will be extended. An `on_match` callback will be
            overwritten.
        
            Parameters
            ----------
            match_id : str
                An ID for the thing you’re matching.

            patterns : List[List[Dict[str, Any]]]
                Match pattern. A pattern consists of a list of dicts, where
                each dict describes a token. 

            on_match : Optional[Callable[['MatcherBase', Doc, int, List[tuple], Any]]], default=None
                Callback function to act on matches. Takes the arguments
                `matcher`, `doc`, `i` and `matches`.

            greedy : Optional[Literal["FIRST", "LONGEST"]], default=None
                Optional filter for greedy matches. Can either be "FIRST" or
                "LONGEST". 
            """
        # Add pattern to custom patterns
        self.patterns[match_id].extend(patterns)

        # Loop over added patterns
        for pattern in patterns:
            # Get current checklist item
            i = len(self.checklist[match_id])
            self.checklist[match_id].append(len(pattern))

            # Loop over tokens in pattern
            for j, token in enumerate(pattern):
                # Add pattern to matcher
                self._matcher.add(
                    f"{match_id}_{i}_{j}", [[token]], greedy=greedy
                )
