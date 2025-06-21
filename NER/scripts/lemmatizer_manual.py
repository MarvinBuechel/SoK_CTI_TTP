import argformat
import argparse
import pandas as pd
import spacy
from spacy_extensions.pipeline import *
from tqdm import tqdm

def parse_args():
    """Parse arguments from command line."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description     = 'Create default config for lemmatizer exceptions from ATT&CK concepts',
        formatter_class = argformat.StructuredFormatter,
    )

    # Add arguments
    parser.add_argument('pipeline', help='pipeline for which to create exceptions')
    parser.add_argument('csv'     , help='path to output file')

    # Parse arguments and return
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Create NLP pipeline
    nlp = spacy.load(args.pipeline)
    # Disable entity mergin
    nlp.disable_pipe('merge_entities')

    ########################################################################
    #                         Load ATT&CK concepts                         #
    ########################################################################

    # Load ATT&CK framework
    attack = nlp.get_pipe('matcher_mitre_attack').attack

    # Get attack concepts (same as matcher_mitre_attack)
    concepts = filter(
        lambda concept: (concept.get('identifier') or '\0')[0] in {
            'T', # Tactics and techniques
            'S', # Software
            'G', # Groups
            'M', # Mitigations
            'D', # Data sources
        },
        attack.concepts
    )

    # Create dictionary from concepts
    dictionary = dict()

    for concept in concepts:
        # Get identifier
        identifier = concept.get('identifier')

        # Add identifier
        if identifier not in dictionary:
            dictionary[identifier] = list()

        # Add names
        names = (
            [concept.get('name')] +
            concept.get('aliases', list()) +
            concept.get('x_mitre_aliases', list())
        )

        # Assert None is not found
        assert None not in names, f"{identifier} contains None"

        # Add names
        dictionary[identifier].extend(names)

    ########################################################################
    #                       Process ATT&CK concepts                        #
    ########################################################################

    # Initialise result
    result = {
        'ATT&CK'      : list(),
        'full_name'   : list(),
        'token'       : list(),
        'lemma'       : list(),
        'lemma_manual': list(),
        'pos'         : list(),
        'pos_manual'  : list(),
        'notes'       : list(),
        'related'     : list(),
    }

    # Loop over entire dictionary
    for identifier, names in tqdm(sorted(dictionary.items())):
        # Get all names
        for name in names:
            # Process name
            doc = nlp(name.lower())

            # Loop over all found tokens
            for token in doc:
                # Add to result
                result['ATT&CK'   ].append(identifier)
                result['full_name'].append(doc.text)
                result['token'    ].append(token.text)
                result['lemma'    ].append(token.lemma_)
                result['pos'      ].append(token.pos_)
                result['notes'    ].append('')
                result['related'  ].append('')

                # Add manual pos and lemma based on identifier
                if identifier.startswith('G') or identifier.startswith('S'):
                    result['lemma_manual'].append(token.lemma_)
                    result['pos_manual'  ].append('PROPN')
                else:
                    result['lemma_manual'].append('')
                    result['pos_manual'  ].append('')


    # Write to csv
    df = pd.DataFrame(result)
    df = df.drop_duplicates(ignore_index=True)
    df.to_csv(args.csv, index=None)
    

if __name__ == '__main__':
    main()