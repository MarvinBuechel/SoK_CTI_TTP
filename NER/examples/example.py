# Imports
import spacy
from spacy_extensions.pipeline import MatcherRegex
from spacy_extensions.pipeline import MatcherTrie

def main():
    # Create pipeline
    nlp = spacy.load('en_core_web_sm')

    # Add pipes
    matcher_regex = nlp.add_pipe('matcher_regex')
    matcher_trie  = nlp.add_pipe('matcher_trie')

    # Train/configure pipes
    matcher_regex.add_regex('hex', '0x[0-9a-fA-F]+')
    matcher_trie .fit(
        sequences = ['matching', 'string'],
        labels    = ['string-match', 'string-match'],
    )

    # Run pipeline on document
    document = nlp(
        "This is an example document. "
        "Here, we give an example of using a regular expression: 0xdeadbeef. "
        "Or we can simply use Trie matching to look for strings. "
    )

    # Show entities
    print(document.ents)

if __name__ == "__main__":
    main()
