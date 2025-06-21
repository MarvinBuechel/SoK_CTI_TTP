import itertools
import os
from pathlib import Path
import argformat
import argparse
import numpy as np
import pandas as pd
import spacy
from spacy_embeddings.pipeline.synonyms.utils import flatten_pos
from spacy_extensions.pipeline import *
from spacy_embeddings.pipeline import *

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description     = 'Collect synonyms from a given trained pipeline',
        formatter_class = argformat.StructuredFormatter,
    )

    # Optional arguments
    parser.add_argument('pipeline', help='pipeline to use for finding synonyms')
    parser.add_argument('outdir'  , help='output directory in which to store synonyms')
    parser.add_argument('--n', type=int, default=5, help='number of synonyms to find per word')
    parser.add_argument('--threshold', type=float, help='threshold to use for finding synonyms')
    
    # Parse arguments
    args = parser.parse_args()

    # Prepare outfile
    args.outdir = Path(args.outdir).absolute()

    ########################################################################
    #                          Configure pipeline                          #
    ########################################################################

    # Setup pipeline
    nlp = spacy.load(args.pipeline)

    # Get synonym finder pipe
    synonym_finder = nlp.get_pipe('synonym_finder')

    # Include 1 more to n, because the token itself will always be found as a synonym
    args.n = args.n+1

    # Configure synonym finder
    synonym_finder.n         = args.n
    synonym_finder.threshold = args.threshold

    # Get ATT&CK framework for description lookup
    attack = nlp.get_pipe('matcher_mitre_attack').attack

    ########################################################################
    #                   Special synonyms (IOC + ATT&CK)                    #
    ########################################################################
    
    # Prepare output directory
    outdir = args.outdir / 'special'
    os.makedirs(outdir, exist_ok=True)

    # Get embeddings and labels of special tokens
    embeddings, labels, counts = flatten_pos(synonym_finder.data)
    mask = np.asarray([
        any(l.startswith('[') and l.endswith(']') for l in label)
        for label in labels
    ])
    embeddings = embeddings[mask]
    labels     = labels    [mask]
    counts     = counts    [mask]

    # Set required frequencies and poss
    frequencies = [5, 10, 20, 50, 100, 200, 500]
    poss        = ['NOUN', 'PROPN', 'VERB']

    # Compute synonyms for different minimum frequencies
    for frequency in frequencies:
        # Initialise result
        result = {
            'token'      : list(),
            'pos'        : list(),
            'count'      : list(),
            'rank'       : list(),
            'synonym'    : list(),
            'description': list(),
        }


        # Parts of speech that are of interest
        for pos in poss:
            # Get predictions
            predictions = synonym_finder.predict(
                embeddings,
                pos       = [pos] * embeddings.shape[0],
                frequency = frequency,
            )

            # Loop over each prediction
            for prediction, label, count in zip(predictions, labels, counts):
                # Get label
                assert len(label) == 1, f"Multiple labels: {label}"
                label = list(label)[0]

                # Get description
                description = attack.get(label[1:-1], {}).get('name')

                # Fill result
                for rank, synset in enumerate(prediction):
                    # Skip unfound synonyms
                    if synset is None: continue

                    for synonym in synset:
                        if synonym != label:
                            result['pos'        ].append(pos)
                            result['token'      ].append(label)
                            result['count'      ].append(count)
                            result['rank'       ].append(rank)
                            result['synonym'    ].append(synonym)
                            result['description'].append(description)

        # Transform to dataframe
        result = pd.DataFrame(result)
        result = result.drop_duplicates()
        result = result.sort_values(['token', 'pos'])
        # Save output to file
        result.to_csv(outdir / f'frequency_{frequency}.csv', index=False)

    ########################################################################
    #                          Find all synonyms                           #
    ########################################################################

    # Prepare output directory
    outdir = args.outdir / 'generic'
    os.makedirs(outdir, exist_ok=True)

    # Set required frequencies and poss
    frequencies = [5, 10, 20, 50, 100, 200, 500]
    poss        = ['NOUN', 'PROPN', 'VERB']

    # Compute synonyms for different minimum frequencies
    for frequency in frequencies:
        # Initialise result
        result = {
            'token'      : list(),
            'pos'        : list(),
            'count'      : list(),
            'rank'       : list(),
            'synonym'    : list(),
        }

        # Loop over all POS tags and corresponding embeddings
        for pos, data in synonym_finder.data.items():
            # Unpack data
            embeddings = data['X']
            labels     = data['y']
            counts     = data['count']

            # Get data of given frequency
            mask = counts >= frequency
            embeddings = embeddings[mask]
            labels     = labels    [mask]
            counts     = counts    [mask]

            # Print progress
            print(f"Computing {pos} ({embeddings.shape[0]} samples)...")

            # Get synonym predictions
            predictions = synonym_finder.predict(
                X         = embeddings,
                pos       = [pos]*embeddings.shape[0],
                frequency = frequency,
            )

            # Ensure prediction is correct
            assert (labels.shape[0], args.n) == predictions.shape

            # Loop over each prediction
            for prediction, label, count in zip(predictions, labels, counts):
                # Get label
                assert len(label) == 1, f"Multiple labels: {label}"
                label = list(label)[0]

                # Fill result
                for rank, synset in enumerate(prediction):
                    # Skip unfound synonyms
                    if synset is None: continue

                    for synonym in synset:
                        if synonym != label:
                            result['pos'        ].append(pos)
                            result['token'      ].append(label)
                            result['count'      ].append(count)
                            result['rank'       ].append(rank)
                            result['synonym'    ].append(synonym)

        # Get result as pandas dataframe
        result = pd.DataFrame(result)
        result = result.drop_duplicates()
        result = result.sort_values(['token', 'pos'])

        # Write to output file
        result.to_csv(outdir / f'frequency_{frequency}.csv', index=False)

if __name__ == '__main__':
    main()
