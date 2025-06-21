import copy
import json
import re
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span
from spacy_extensions.utils import SpacyCallable, SpacySerializable
from spacy_extensions.utils.tree import min_subtree
from typing import Any, Dict, List, Optional
from typing import Iterable

@spacy.Language.factory("matcher_sentence", assigns=['Span._.sentence_matches'])
class SentenceMatcher(SpacyCallable, SpacySerializable):

    def __init__(self, nlp, name):
        """Match on given sentences."""
        # Initialise nlp
        self.nlp = nlp
        self.name = name

        # Initialise matcher
        self.matcher = Matcher(self.nlp.vocab)
        self.patterns = list()
        self.checklist = dict()

        Span.set_extension(
            name = 'sentence_matches',
            default = list(),
        )

        # Initialise regex
        self.regex = re.compile(r'^(.*)_(\d+)_(\d+)$')

    ########################################################################
    #                              Detection                               #
    ########################################################################
    
    def __call__(self, doc: Doc) -> Doc:
        # If matcher was not fitted, return doc
        if len(self.matcher) == 0: return doc

        # Initialise spans
        spans_all = list()

        # Perform matching per sentence
        for sentence in doc.sents:
            spans = list()

            # Set found matches
            matches = copy.deepcopy(self.checklist)

            # Match sentence
            for label, index, _ in self.matcher(sentence):
                # Unpack label
                match = self.regex.match(self.nlp.vocab[label].text)
                label, i, j = match.group(1), match.group(2), match.group(3)
                i, j = int(i), int(j)

                # Add match
                matches[label][i][j].add(sentence[index].i)

            # Check which matches have been completed
            for label, patterns in matches.items():
                for pattern in filter(None, patterns):
                    while all(len(token) for token in pattern):
                        
                        # Greedy search for minimum subtrees
                        span = min_subtree(
                            sentence = sentence,
                            includes = [[doc[i] for i in part] for part in pattern],
                        )

                        # Stop if we couldn't find a new span
                        if span is None: break

                        # Set span label
                        span.label_ = label

                        # Append span
                        spans.append(span)
                        # Remove used patterns
                        pattern = [{
                            index for index in part
                            if index < span.start or index >= span.end
                            } for part in pattern
                        ]

            # Set sentence matches
            sentence._.sentence_matches = spans
            spans_all.extend(spans)

        # Set document entities
        doc.set_ents(
            spacy.util.filter_spans(
                tuple(filter(None, spans_all)) +
                doc.ents
            )
        )

        # Return doc
        return doc

    ########################################################################
    #                             Add patterns                             #
    ########################################################################

    def add(
            self,
            match_id: str,
            patterns: List[List[Dict[str, Any]]],
            greedy: Optional[str] = None,
        ) -> None:
        """Add a pattern for a match ID."""
        # Save added patterns
        self.patterns.append([match_id, patterns, greedy])

        if match_id not in self.checklist:
            self.checklist[match_id] = list()

        # Loop over all patterns
        for pattern in patterns:
            # Add pattern to checklist
            i = len(self.checklist[match_id])
            self.checklist[match_id].append(list())

            # Loop over tokens in pattern
            for token in pattern:
                # Add token to checklist
                j = len(self.checklist[match_id][i])
                self.checklist[match_id][i].append(set())

                # Add pattern to matcher
                self.matcher.add(
                    f"{match_id}_{i}_{j}",
                    [[token]],
                    greedy = greedy,
                )

    ########################################################################
    #                             I/O methods                              #
    ########################################################################

    def to_bytes(self, exclude: Optional[Iterable[str]] = None) -> bytes:
        """Return a bytes representation of SentenceMatcher."""
        return json.dumps(self.patterns).encode('utf-8')


    def from_bytes(self, data: bytes, exclude: Optional[Iterable[str]] = None) -> SpacySerializable:
        """Load SentenceMatcher from bytes representation."""
        # Loop over all trained patterns
        for match_id, patterns, greedy in json.loads(data.decode('utf-8')):
            # Fit model
            self.add(match_id, patterns, greedy)

        # Return self
        return self