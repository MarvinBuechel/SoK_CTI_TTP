import json
import argformat
import argparse
import spacy
from nltk.corpus import wordnet as wn
from spacy_extensions.pipeline import *
from spacy_extensions.utils.attack import get_attack_concept_names
from tqdm import tqdm

def parse_args():
    """Parse arguments from command line."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description     = 'Create default config for POS exceptions from ATT&CK concepts',
        formatter_class = argformat.StructuredFormatter,
    )

    # Add arguments
    parser.add_argument('pipeline', help='pipeline for which to create exceptions')
    parser.add_argument('json'    , help='path to output file')

    # Parse arguments and return
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Create NLP pipeline
    nlp = spacy.load(args.pipeline)
    # Disable entity mergin
    nlp.disable_pipe('merge_entities')

    # Load ATT&CK concepts
    dictionary = get_attack_concept_names(nlp)

    ########################################################################
    #                        Create exception rules                        #
    ########################################################################

    # Initialise result
    result = {}

    # Loop over entire dictionary
    for identifier, names in tqdm(sorted(dictionary.items())):

        # Set software and groups to PROPN
        if identifier.startswith('G') or identifier.startswith('S'):
            pos = "PROPN"

            # Get all names
            for name in names:
                # Process name
                doc = nlp(name.lower())

                # Create name
                name = '_'.join(token.text for token in doc)

                # Check for duplicates
                assert result.get(name, pos) == pos

                # Do not override POS if it is also a regular word
                if wn.lemmas(name): continue

                # Set result
                result[name] = pos

    # Write result to outfile
    with open(args.json, 'w') as outfile:
        json.dump(result, outfile, indent=4, sort_keys=True)
    

if __name__ == '__main__':
    main()