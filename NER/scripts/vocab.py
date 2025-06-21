# Ignore tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argformat
import argparse
import spacy
from spacy_extensions.pipeline import *
from spacy_extensions.utils.pipeline import nlp2preprocessed
from spacy_extensions.utils.preprocessing import load_files_docbin, load_files_txt
from spacy_extensions.vocab import create_vocab_file

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description     = 'Create vocab for documents',
        formatter_class = argformat.StructuredFormatter,
    )

    # Optional arguments
    parser.add_argument('pipeline', help='pipeline to use for creating vocab')
    parser.add_argument('outfile' , help='output file in which to store vocab')
    parser.add_argument('files'   , nargs='+', help='files to from which to extract vocab')
    parser.add_argument('--preprocessed'   , action='store_true' , help='if set, assume files are DocBins preprocessed with given pipeline')
    parser.add_argument('--exclude-special', action='store_false', help='if set, create vocab by excluding special tokens')

    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                          Configure pipeline                          #
    ########################################################################

    # Setup pipeline
    nlp = spacy.load(args.pipeline)
    
    # Remove preprocessing components, if already preprocessed
    if args.preprocessed:
        nlp = nlp2preprocessed(nlp)

    ########################################################################
    #                             Create vocab                             #
    ########################################################################

    # Get file loader
    if args.preprocessed:
        files = load_files_docbin(args.files, nlp.vocab)
    else:
        files = load_files_txt(args.files)

    # Create vocabulary file
    create_vocab_file(
        outfile   = args.outfile,
        documents = map(nlp, files),
        escape    = True,
        special   = args.exclude_special,
        verbose   = True,
    )


if __name__ == '__main__':
    main()