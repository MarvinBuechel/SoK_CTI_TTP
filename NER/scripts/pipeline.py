# Ignore tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argformat
import argparse
import os
from pathlib import Path
from spacy_embeddings.pipeline.embedders.types import TypeEmbedder
from typing import get_args

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description     = 'Create a full pipeline from a given set of documents',
        formatter_class = argformat.StructuredFormatter,
    )

    # Optional arguments
    parser.add_argument('name'        , help='name of pipe')
    parser.add_argument('base'        , help='base pipe to use')
    parser.add_argument('cti'         , help='path to cti')
    parser.add_argument('files'       , nargs='+', help='files to use as input')
    parser.add_argument(
        '--embedders',
        nargs   = '+',
        default = list(get_args(TypeEmbedder)),
        choices = get_args(TypeEmbedder),
        help    = 'embedder to use in pipeline(s)',
    )
    
    
    # Parse arguments
    args = parser.parse_args()

    ########################################################################
    #                             Create pipes                             #
    ########################################################################

    # Get path to spacy-extensions directory
    base_path = Path(__file__).absolute().parent.parent

    # Output files and directories
    output_base            = Path(args.name).absolute()
    output_pipes           = output_base   / 'pipelines'
    output_models          = output_base   / 'models'
    output_models_embedder = output_models / 'embedders'
    output_models_sf       = output_models / 'synonym_finders'
    output_preprocessed    = output_base   / 'preprocessed'

    # Create output directories
    os.makedirs(output_base           , exist_ok=True)
    os.makedirs(output_pipes          , exist_ok=True)
    os.makedirs(output_models         , exist_ok=True)
    os.makedirs(output_models_embedder, exist_ok=True)
    os.makedirs(output_models_sf      , exist_ok=True)
    os.makedirs(output_preprocessed   , exist_ok=True)

    # Create base pipe
    print("\nCreating base pipe:")
    command = (
        f"python3 {base_path / 'pipelines'} " # Call pipeline script
        f"{output_pipes / 'base'} "           # Pipeline name to create
        f"--base {args.base} "                # Set corresponding base pipeline
        f"--cti {args.cti} "                  # Set CTI path
    )
    print(command[:500])
    error = os.system(command)
    if error:
        raise ValueError(f"System exit: {error}")

    # Preprocess files
    print("\nPreprocessing files:")
    command = (
        f"python3 {base_path / 'scripts' / 'preprocess.py'} " # Call preprocess script
        f"{output_pipes / 'base'} "                           # Use base pipeline for preprocessing
        f"{output_preprocessed} "                             # Store files in given output directory
        f"{' '.join(args.files)}"                             # Files to preprocess
    )
    print(command[:500])
    error = os.system(command)
    if error:
        raise ValueError(f"System exit: {error}")

    # Collect preprocessed files
    preprocessed_files = str(Path(output_preprocessed) / '*')

    # Create vocab
    print("\nCreating vocab:")
    command = (
        f"python3 {base_path / 'scripts' / 'vocab.py'} " # Call vocab script
        f"{output_pipes / 'base'} "                      # Use base pipeline
        f"{output_models / 'vocab'} "                    # Output into models/vocab
        f"{preprocessed_files} "                         # Use preprocessed files
        "--preprocessed"                                 # Files have been preprocessed
    )
    print(command[:500])
    error = os.system(command)
    if error:
        raise ValueError(f"System exit: {error}")

    # Loop over all embedders
    for embedder in args.embedders:

        # Create untrained embedder pipe(s)
        print(f"\nInitialising untrained {embedder} pipe:")
        command = (
            f"python3 {base_path / 'pipelines'} "   # Call pipeline script
            f"{output_pipes / embedder}_untrained " # Embedder output
            f"--base {args.base} "                  # Base pipeline to use
            f"--cti {args.cti} "                    # CTI location
            f"--vocab {output_models / 'vocab'} "   # Vocab file
            f"--embedder {embedder}"                # Embedder to use
        )
        print(command[:500])
        error = os.system(command)
        if error:
            raise ValueError(f"System exit: {error}")

        # Train embedder
        print(f"\nTraining {embedder} embedder:")
        command = (
            f"python3 {base_path / 'scripts' / 'embedder.py'} " # Call embedder script
            f"{output_pipes / embedder}_untrained "             # Embedder pipeline
            f"{output_models_embedder / embedder}.model "       # Embedder model output
            f"{preprocessed_files} "                            # Use preprocessed files
            "--preprocessed "                                   # Files have been preprocessed
            "--epochs 10 "                                      # Only train for 10 epochs to speed up process
            f"--batch-size {32 if embedder in ['bert', 'elmo'] else 128}" # Batch size is 32 for sentences, or 128 for tokens
        )
        print(command[:500])
        error = os.system(command)
        if error:
            raise ValueError(f"System exit: {error}")

        # Create trained embedder pipe(s)
        print(f"\nInitialising trained {embedder} pipe:")
        command = (
            f"python3 {base_path / 'pipelines'} "                        # Call pipeline script
            f"{output_pipes / embedder}_trained "                        # Trained embedder pipeline
            f"--base {args.base} "                                       # Base pipeline to use
            f"--cti {args.cti} "                                         # CTI location
            f"--vocab {output_models / 'vocab'} "                        # Vocab file
            f"--embedder {embedder} "                                    # Embedder to use
            f"--embedder-path {output_models_embedder / embedder}.model" # Trained embedder to use
        )
        print(command[:500])
        error = os.system(command)
        if error:
            raise ValueError(f"System exit: {error}")

        # Train synonym finder
        print("\nTraining synonym finder:")
        command = (
            f"python3 {base_path / 'scripts' / 'synonym_finder.py'} " # Call synonym_finder script
            f"{output_pipes / embedder}_trained "                     # Use the trained embedder pipeline
            f"{output_models_sf / embedder}.sf "                      # Synonym finder model output
            f"{preprocessed_files} "                                  # Use preprocessed files
            "--preprocessed"                                          # Files have been preprocessed
        )
        print(command[:500])
        error = os.system(command)
        if error:
            raise ValueError(f"System exit: {error}")

        # Create trained embedder pipe(s)
        print(f"\nCreating final {embedder} pipeline:")
        command = (
            f"python3 {base_path / 'pipelines'} "                         # Call pipeline script
            f"{output_pipes / embedder} "                                 # Trained embedder pipeline
            f"--base {args.base} "                                        # Base pipeline to use
            f"--cti {args.cti} "                                          # CTI location
            f"--vocab {output_models / 'vocab'} "                         # Vocab file
            f"--embedder {embedder} "                                     # Embedder to use
            f"--embedder-path {output_models_embedder / embedder}.model " # Trained embedder to use
            f"--synonym-finder-path {output_models_sf / embedder}.sf"     # Trained synonym finder to use
        )
        print(command[:500])
        error = os.system(command)
        if error:
            raise ValueError(f"System exit: {error}")


if __name__ == '__main__':
    main()