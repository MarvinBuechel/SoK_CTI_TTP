import argformat
import argparse
import spacy
from spacy_extensions.pipeline import *
from spacy_embeddings.pipeline import *
from spacy_extensions.utils.pipeline import nlp2preprocessed
from spacy_extensions.utils.preprocessing import load_files_docbin, load_files_txt
from spacy_embeddings.pipeline.synonyms.utils import get_embedding_labels

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description     = 'Train synonym finder using given documents',
        formatter_class = argformat.StructuredFormatter,
    )

    # Optional arguments
    parser.add_argument('pipeline', help='pipeline to use for training synonym finder')
    parser.add_argument('outfile' , help='output file in which to store synonym finder')
    parser.add_argument('files'   , nargs='+', help='files with which to train synonym finder')
    parser.add_argument('--preprocessed'   , action='store_true' , help='if set, assume files are DocBins preprocessed with given pipeline')
    
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

    # Get synonym finder pipe
    synonym_finder = nlp.get_pipe('synonym_finder')

    ########################################################################
    #                         Train Synonym Finder                         #
    ########################################################################

    # Get file loader
    if args.preprocessed:
        files = load_files_docbin(args.files, nlp.vocab)
    else:
        files = load_files_txt(args.files)

    # Get embeddings and labels of documents after being run through pipeline
    X, y, pos = get_embedding_labels(
        docs = map(nlp, files),
        verbose = True,
    )

    # Train embedder
    synonym_finder.fit(
        X       = X,
        y       = y,
        pos     = pos,
        verbose = True,
    )

    # Write embedder to output file
    synonym_finder.to_disk(args.outfile)


if __name__ == '__main__':
    main()