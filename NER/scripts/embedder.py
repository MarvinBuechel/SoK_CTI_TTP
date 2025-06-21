import argformat
import argparse
import spacy
import torch
from spacy_extensions.pipeline import *
from spacy_embeddings.pipeline import *
from spacy_extensions.utils.pipeline import nlp2preprocessed
from spacy_extensions.utils.preprocessing import load_files_docbin, load_files_txt
from spacy_extensions.vocab import create_vocab_file

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description     = 'Train embedder using given documents',
        formatter_class = argformat.StructuredFormatter,
    )

    # Optional arguments
    parser.add_argument('pipeline', help='pipeline to use for training embedder')
    parser.add_argument('outfile' , help='output file in which to store embedder')
    parser.add_argument('files'   , nargs='+', help='files with which to train embedder')
    parser.add_argument('--preprocessed'   , action='store_true' , help='if set, assume files are DocBins preprocessed with given pipeline')
    
    # Embedder arguments
    embedder_group = parser.add_argument_group("Embedder")
    embedder_group.add_argument('--embedder'  , help='train given embedder in pipeline, if not given, automatically detect embedder')
    embedder_group.add_argument('--epochs'    , type=int, default=100, help='number of epochs to train with')
    embedder_group.add_argument('--batch-size', type=int, default=128, help='size of batch to train with')
    
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
    #                      Find embedder in pipeline                       #
    ########################################################################

    # Set embedder through argument
    if args.embedder:
        embedder = nlp.get_pipe(args.embedder)

    # Automatically detect embedder
    else:
        # Initialise embedder
        embedder = list()
        
        # Loop over all pipes in pipeline
        for name, pipe in nlp.pipeline:
            # Check for embedder
            if name.startswith('embedder_'):
                embedder.append(pipe)

        # Check whether pipeline contains a single embedder
        if len(embedder) == 0:
            raise ValueError("Pipeline does not contain embedder")
        elif len(embedder) > 1:
            raise ValueError("Pipeline contains multiple embedders")

        embedder = embedder[0]

    ########################################################################
    #                            Train embedder                            #
    ########################################################################

    # Get file loader
    if args.preprocessed:
        files = load_files_docbin(args.files, nlp.vocab)
    else:
        files = load_files_txt(args.files)

    # Run model on GPU if available
    if torch.cuda.is_available():
        embedder.model.to('cuda')

    # Train embedder
    embedder.model.fit(
        X          = map(nlp, files),
        epochs     = args.epochs,
        batch_size = args.batch_size,
        verbose    = True,
        save       = args.outfile + '.{epoch}',
    )

    # Write embedder to output file
    embedder.model.to_disk(args.outfile)


if __name__ == '__main__':
    main()