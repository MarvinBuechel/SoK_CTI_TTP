# Ignore tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argformat
import argparse
import os
import spacy
import warnings
from pathlib import Path
from spacy_extensions.pipeline import *
from spacy_extensions.utils.pipeline import nlp2base
from tqdm import tqdm

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description     = 'Preprocessor for documents',
        formatter_class = argformat.StructuredFormatter,
    )

    # Optional arguments
    parser.add_argument('pipeline', help='pipeline to use for preprocessing')
    parser.add_argument('outdir'  , help='output directory to store preprocessed files')
    parser.add_argument('files'   , nargs='+', help='files to preprocess')

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                          Configure pipeline                          #
    ########################################################################

    # Setup pipeline
    nlp = spacy.load(args.pipeline)
    # Get base pipeline from nlp
    nlp = nlp2base(nlp)

    ########################################################################
    #                     Initialize output directory                      #
    ########################################################################

    # Create output directory
    outdir = Path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    ########################################################################
    #                         Preprocess documents                         #
    ########################################################################

    # Loop over files
    for path in tqdm(args.files, desc="Preprocessing"):
        # Transform path to Path
        path = Path(path)
        # Get output path
        outfile = outdir / path.stem

        # Ensure output path does not exist yet
        if os.path.isfile(outfile):
            warnings.warn(f"'{outfile}' already exists, skipping...")
            continue

        # Read document
        with open(path) as infile:
            # Process document
            doc = nlp(infile.read())
            
        # Save document to disk
        spacy.tokens.DocBin(docs = [doc]).to_disk(outfile)


if __name__ == '__main__':
    main()