import json
import warnings
import argformat
import argparse
import spacy
from py_attack import ATTACK, DomainTypes
from nltk.corpus import wordnet as wn
from spacy_extensions.pipeline import *
from spacy_extensions.utils.attack import get_attack_concept_names, get_names_from_ATTACK
from tqdm import tqdm
from typing import get_args

def parse_args():
    """Parse arguments from command line."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description     = 'Create default config for lemmatizer exceptions from ATT&CK concepts',
        formatter_class = argformat.StructuredFormatter,
    )

    # Add arguments
    parser.add_argument('pipeline', help='pipeline for which to create exceptions')
    parser.add_argument('json'    , help='path to output file')
    parser.add_argument('--files', nargs='+', default=[], help='search for relevant inflictions in files')
    parser.add_argument('--cti', help='path to ATT&CK cti')
    parser.add_argument('--domains', choices=get_args(DomainTypes), default=['enterprise'], help='path to ATT&CK cti')

    # Parse arguments and return
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Create NLP pipeline
    nlp = spacy.load(args.pipeline)
    # Disable entity mergin
    if nlp.has_pipe('merge_entities'):
        nlp.disable_pipe('merge_entities')

    # Load ATT&CK concepts
    if args.cti:
        attack = ATTACK.load(
            path    = args.cti,
            domains = args.domains,
        )
        dictionary = get_names_from_ATTACK(attack)
    else:
        dictionary = get_attack_concept_names(nlp)

    ########################################################################
    #                        Create exception rules                        #
    ########################################################################

    # Initialise result
    result = {
        "propn": dict(),
        "noun": dict(),
        "verb": dict(),
        "adj": dict(),
        "adv": dict()
    }

    # Mapping from wordnet to generic POS tag
    wn_pos = {
        'a': 'adj',
        's': 'adj', # We map satellite adjectives to adjectives
        'r': 'adv',
        'n': 'noun',
        'v': 'verb',
    }

    # Loop over entire dictionary
    for identifier, names in tqdm(sorted(dictionary.items())):
        # Get all names
        for name in names:
            # Process name
            doc = nlp(name.lower())

            # Loop over all found tokens
            for token in doc:
                # Get related lemmas in wordnet
                lemmas = wn.lemmas(token.text) + wn.lemmas(token.lemma_)
                related = [lemma for lemma in lemmas]
                for lemma in lemmas:
                    related += lemma.derivationally_related_forms()

                # Get unique (lemma, pos) tuples
                related = set([
                    (entry.name().lower(), wn_pos[entry.synset().pos()])
                    for entry in related
                ])

                # Cases not in wordnet
                if token.text == 'lateral':
                    related.add(('laterally', 'adj'))

                # Add original related mapping
                related.add((token.lemma_.lower(), token.pos_.lower()))

                # Create target lemma based on following scheme:
                # - shortest
                # - pos == verb
                # - pos
                # - alphabetical
                targets = sorted(related, key = lambda x: x[0])
                targets = sorted(targets, key = lambda x: x[1])
                targets = sorted(targets, key = lambda x: x[1] != "verb")
                targets = sorted(targets, key = lambda x: len(x[0]))
                target = list(targets)[0]
                target = (target[0].lower(), target[1])

                # Exceptions because of incorrect token POS tagging
                if target[0] == 'spider' and target[1] == 'verb':
                    target = ('spider', "noun")
                elif target[0] == 'window' or target[0] == 'windows':
                    target = ("window", "noun")
                elif target[0] == 'persona' and target[1] == 'noun':
                    target = ("impersonate", "verb")
                elif target[0] == 'startup' and target[1] == 'noun':
                    target = ("startup", "verb")
                elif target[0] == 'os':
                    target = ("os", "noun")
                elif target[0] == 'arp':
                    target = ("arp", "noun")
                elif target[0] == 'preside':
                    target = ("president", "noun")
                elif target[0] == 'cryptography':
                    target = ("crypto", "noun")

                # Add to list of exceptions
                for lemma, pos in related:
                    # Skip pos that is not in result
                    if pos not in result: continue

                    # Check for duplicate target mismatches
                    if result[pos].get(lemma, target)[0] != target[0]:

                        # Select shortest as target
                        target_ = min(
                            [result[pos].get(lemma, target), target],
                            key = lambda x: len(x[0])
                        )

                        # Raise warning
                        warnings.warn(
                            f"Target mismatch for '{lemma}' ({pos}): "
                            f"'{result[pos][lemma]}' != '{target}'. "
                            f"Default to {target_}"
                        )

                        # Set target
                        target = target_

                    # Add result
                    result[pos][lemma] = target

    # Transform result to correct form
    for pos, exceptions in result.items():
        for input, (exception, _) in exceptions.items():
            result[pos][input] = [exception]

    ########################################################################
    #              Include all inflictions found in documents              #
    ########################################################################

    # Disable unnecessary parts of pipeline
    for pipe, _ in nlp.pipeline:
        nlp.disable_pipe(pipe)
    nlp.enable_pipe('tok2vec')
    nlp.enable_pipe('tagger')
    nlp.enable_pipe('attribute_ruler')

    # Mapping from wordnet to generic POS tag
    wn_pos = {
        'ADJ': 'a',
        'ADV': 'r',
        'NOUN': 'n',
        'VERB': 'v',
    }

    # Loop over all files
    for file in tqdm(args.files, desc='inflictions'):
        with open(file) as infile:
            data = infile.read()

        # Loop over all tokens in file
        for token in nlp(data):
            # Set target
            target = None

            # Skip unknown token pos
            if token.pos_ not in wn_pos:
                continue
            pos = wn_pos[token.pos_]

            # Its morphin' time
            morphed = wn.morphy(token.text.lower())
            if morphed is None: continue

            # Check all lemmas
            for lemma in wn.lemmas(morphed, pos=pos):
                # Get possible targets
                for pos, values in result.items():
                    # Check if lemma exists
                    if lemma.name().lower() in values:
                        # Set to target
                        assert target is None or target == values[lemma.name().lower()]
                        target = values[lemma.name().lower()]

            # Add newly found infliction
            if target:
                if result[token.pos_.lower()].get(token.text.lower(), target) != target:
                    # Select shortest as target
                        target_ = min([
                            result[token.pos_.lower()].get(token.text.lower(), target),
                            target
                            ], key = lambda x: len(x[0])
                        )

                        # Raise warning
                        warnings.warn(
                            f"Duplicate target [{token.text.lower()}]: "
                            f"'{result[token.pos_.lower()].get(token.text.lower(), target)}'"
                            f" != '{target}', selecting {target_}"
                        )

                        # Set target
                        target = target_

                result[token.pos_.lower()][token.text.lower()] = target

    # Write to output file
    with open(args.json, 'w') as outfile:
        # Get data as json
        json.dump(result, outfile, indent=4, sort_keys=True)

    

if __name__ == '__main__':
    main()