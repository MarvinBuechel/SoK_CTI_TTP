# Ignore tensorflow warnings
from cmath import inf
import json
import os

from spacy_extensions.utils.exceptions import add_lemmatizer_exceptions, add_pos_exceptions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Imports
import warnings
import argparse
import spacy
from pathlib import Path
from py_attack.types import DomainTypes
from spacy_embeddings.pipeline.embedders.types import TypeEmbedder
from spacy_extensions.pipeline import *
from spacy_embeddings.pipeline.synonyms.types import SupportedMetrics
from typing import Dict, List, Optional, Tuple, Union, get_args

from spacy_extensions.utils.iocs import IOCs

################################################################################
#                            Pipeline configuration                            #
################################################################################

def build(
        # Base configuration
        base: str = 'en_core_web_sm',

        # Exceptions
        exceptions_pos: Optional[str] = None,
        exceptions_lemma: Optional[str] = None,

        # matcher_mitre_attack configuration
        attack_path: Optional[str] = None,
        attack_domains: List[DomainTypes] = ['enterprise'],

        # vocab_mapper configuration
        vocab_path: Optional[str] = None,
        vocab_freq: int = 5,

        # Contextualizer configuration
        window: Union[int, Tuple[int, int]] = 5,

        # Embedder configuration
        embedder     : Optional[TypeEmbedder] = None,
        dim_embedding: int   = 300,
        dim_hidden   : int   = 256,
        max_context  : int   = 128,
        train_ratio  : float = 0.15,
        embedder_path: Optional[str] = None,

        # SynonymFinder configuration
        metric             : SupportedMetrics = 'cosine',
        n                  : Optional[int]    = 5,
        threshold          : Optional[float]  = None,
        pos                : bool             = True,
        synonym_finder_path: Optional[str]    = None,

        # Set verbose
        verbose: bool = False,

    ) -> spacy.Language:
    """Build a spacy CTI pipeline based on the given arguments.
    
        Parameters
        ----------
        base : str, default='en_core_web_sm'
            Base language to extend with CTI pipes. See
            https://spacy.io/usage/models for available models.

        Returns
        -------
        nlp : spacy.Language
            CTI pipeline based on given arguments.
        """
    ################################################################
    #                            Checks                            #
    ################################################################

    if vocab_path is None and embedder is not None:
        raise ValueError(
            f"embedder='{embedder}', but no 'vocab_path' was specified. "
            "Embedder requires the 'vocab_path' argument to be specified."
        )

    ################################################################
    #                        Build pipeline                        #
    ################################################################

    # Initialise pipeline
    nlp = spacy.load(base)

    # Add POS exceptions
    if exceptions_pos:
        with open(exceptions_pos) as infile:
            exceptions_pos = json.load(infile)
        nlp = add_pos_exceptions(nlp, exceptions_pos)

    # Add Lemma exceptions
    if exceptions_lemma:
        with open(exceptions_lemma) as infile:
            exceptions_lemma = json.load(infile)
        nlp = add_lemmatizer_exceptions(nlp, exceptions_lemma)

    # Configure NER pipes
    nlp.disable_pipe('ner')

    # Load MatcherIOC and add IOCs
    matcher_ioc = nlp.add_pipe('matcher_ioc')
    matcher_ioc.add_regexes(IOCs)

    # Load MatcherMitreAttack and add ATT&CK
    matcher_attack = nlp.add_pipe('matcher_mitre_attack')
    matcher_attack.from_cti(
        path    = attack_path,
        domains = attack_domains,
    )

    # Set pipe to merge entities
    nlp.add_pipe('merge_entities')

    # If no vocab path is given, return nlp
    if vocab_path is None:
        warnings.warn(
            "\n\nNo vocab_path specified, returning pipeline:\n"
            f"\t- base ({base})\n"
            "\t- ner (disabled)\n"
            "\t- matcher_ioc\n"
            "\t- matcher_mitre_attack\n"
            "\t- merge_entities\n"
        )
        return nlp

    # Configure vocab_mapper
    vocab_mapper = nlp.add_pipe('vocab_mapper')
    vocab_mapper.set_vocab_file(
        path      = vocab_path,
        frequency = vocab_freq,
    )

    # Configure contextualizers
    context_w = nlp.add_pipe('contextualizer_window', config={
        'context': window
    })
    nlp.add_pipe('contextualizer_sentence')

    # If no embedder is given, return nlp
    if embedder is None:
        warnings.warn(
            "\n\nNo embedder specified, returning pipeline:\n"
            f"\t- base ({base})\n"
            "\t- ner (disabled)\n"
            "\t- matcher_ioc\n"
            "\t- matcher_mitre_attack\n"
            "\t- merge_entities\n"
            "\t- vocab_mapper\n"
            "\t- contextualizer_window\n"
            "\t- contextualizer_sentence\n"
        )
        return nlp

    # Configure embedder
    if embedder == 'bert':
        embedder = nlp.add_pipe('embedder_bert', config={
            'dim_input'       : len(vocab_mapper),
            'dim_hidden'      : dim_embedding,
            'max_context'     : max_context,
            'sep_index'       : vocab_mapper.sep_index,
            'pad_index'       : vocab_mapper.pad_index,
            'cls_index'       : vocab_mapper.cls_index,
            'mask_index'      : vocab_mapper.mask_index,
            'train_mask_ratio': train_ratio,
        })
    elif embedder == 'elmo':
        embedder = nlp.add_pipe('embedder_elmo', config={
            'dim_input'       : len(vocab_mapper),
            'dim_hidden'      : dim_embedding // 2,
            'max_context'     : max_context,
            'pad_index'       : vocab_mapper.pad_index,
            'mask_index'      : vocab_mapper.mask_index,
            'train_mask_ratio': train_ratio,
        })
    elif embedder == 'nlm':
        embedder = nlp.add_pipe('embedder_nlm', config={
            'dim_input'    : len(vocab_mapper),
            'context'      : sum(context_w.context),
            'dim_embedding': dim_embedding,
            'dim_hidden'   : dim_hidden,
        })
    elif embedder == 'word2vec-cbow':
        embedder = nlp.add_pipe('embedder_word2vec_cbow', config={
            'dim_input'    : len(vocab_mapper),
            'dim_embedding': dim_embedding,
            'dim_hidden'   : dim_hidden,
        })
    elif embedder == 'word2vec-skipgram':
        embedder = nlp.add_pipe('embedder_word2vec_skipgram', config={
            'dim_input'    : len(vocab_mapper),
            'context'      : sum(context_w.context),
            'dim_embedding': dim_embedding,
            'dim_hidden'   : dim_hidden,
        })

    # Raise error for unknown embedder
    else:
        raise ValueError(
            f"Unknown embedder: '{embedder}'. Should be one of {TypeEmbedder}"
        )

    # Load embedder from path
    if embedder_path:
        embedder = embedder.from_disk(embedder_path)
    else:
        warnings.warn(
            "Creating an untrained embedder. Make sure you train your embedder "
            "before using the pipeline in practice."
        )
        return nlp

    # Configure SynonymFinder
    synonym_finder = nlp.add_pipe('synonym_finder', config={
        'metric'   : metric,
        'n'        : n,
        'threshold': threshold,
        'pos'      : pos,
    })

    # Load synonym_finder from path
    if synonym_finder_path:
        synonym_finder = synonym_finder.from_disk(synonym_finder_path)
    else:
        warnings.warn(
            "Creating an untrained synonym finder. Make sure you train your "
            "synonym finder before using the pipeline in practice."
        )
        return nlp

    # Return pipeline
    return nlp

################################################################################
#                                Show pipeline                                 #
################################################################################

def summary(nlp: spacy.Language) -> None:
    """Print a summary of the nlp pipeline.
    
        Parameters
        ----------
        nlp : spacy.Language
            Pipeline of which to print summary.
        """
    # Get configuration
    components = nlp.config.get('components')

    # Print header
    print("NLP pipeline:")

    # Loop over pipeline
    for name, _ in nlp.pipeline:
        # Get config
        if name not in {
            'attribute_ruler', 'lemmatizer', 'ner',
            'parser', 'senter', 'tagger', 'tok2vec'}:
            config = components.get(name)
            config = ', '.join(
                f'{param}={value}' for param, value in sorted(config.items())
                if param != 'factory'
            )
            config = config or 'N/A'
        else:
            config = 'N/A'

        print(f"  - {name} ({config})")

################################################################################
#                                     Main                                     #
################################################################################

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description = 'Setup SpaCy cti pipeline')

    # Optional arguments
    parser.add_argument(
        'outdir',
        help = 'directory in which to store pipeline',
    )
    parser.add_argument(
        '--base',
        default = 'en_core_web_sm',
        help    = 'base to use for pipeline',
    )

    # CTI arguments
    group_cti = parser.add_argument_group('CTI')
    group_cti.add_argument(
        '--cti',
        required = True,
        help     = 'path to MITRE ATT&CK cti directory',
    )
    group_cti.add_argument(
        '--domains',
        nargs   = '+',
        default = ['enterprise'],
        help    = 'MTIRE ATT&CK domains to include',
    )

    # Vocab arguments
    group_vocab = parser.add_argument_group('Vocab')
    group_vocab.add_argument(
        '--vocab',
        help = 'path to vocab file from which to load vocab',
    )
    group_vocab.add_argument(
        '--frequency',
        type    = int,
        default = 5,
        help    = 'minimum frequency of vocab to be included',
    )

    # Contextualizer arguments
    group_context = parser.add_argument_group("Context")
    group_context.add_argument(
        '--window',
        type    = int,
        default = 5,
        help    = 'context window to use for contextualizer_window',
    )

    # Embedder
    group_embedder = parser.add_argument_group("Embedder")
    group_embedder.add_argument(
        '--embedder',
        choices = get_args(TypeEmbedder),
        help    = 'embedder type to use'
    )
    group_embedder.add_argument(
        '--dim-embedding',
        type    = int,
        default = 300,
        help    = 'dimension of created embeddings'
    )
    group_embedder.add_argument(
        '--dim-hidden',
        type    = int,
        default = 256,
        help    = 'hidden dimension of NN (nlm & word2vec only)'
    )
    group_embedder.add_argument(
        '--max-context',
        type    = int,
        default = 128,
        help    = 'maximum size of context (bert & elmo only)'
    )
    group_embedder.add_argument(
        '--train-ratio',
        type    = float,
        default = 0.15,
        help    = 'ratio of words to mask in training (bert & elmo only)'
    )
    group_embedder.add_argument(
        '--embedder-path',
        help = 'path to pretrained embedder'
    )

    # SynonymFinder arguments
    group_synonym_finder = parser.add_argument_group("SynonymFinder")
    group_synonym_finder.add_argument(
        '--metric',
        choices = get_args(SupportedMetrics),
        default = 'cosine',
        help    = 'distance metric to use for finding similar vectors',
    )
    group_synonym_finder.add_argument(
        '--n',
        type    = int,
        default = 5,
        help    = 'maximum number of synonyms to find',
    )
    group_synonym_finder.add_argument(
        '--threshold',
        type    = float,
        help    = 'maximum distance for which vectors are concidered synonymous',
    )
    group_synonym_finder.add_argument(
        '--no-pos',
        action = 'store_true',
        help   = 'if set, do not use part-of-speech for synonym finding',
    )
    group_synonym_finder.add_argument(
        '--synonym-finder-path',
        help    = 'path from which to load trained synonym finder',
    )

    # Parse arguments
    args = parser.parse_args()

    # Prepare ATT&CK CTI path
    attack_path = str(Path(args.cti) / '{domain}-attack' / '{domain}-attack.json')

    # build pipe based on given arguments
    nlp = build(
        # Base pipeline
        base = args.base,

        # CTI arguments
        attack_path    = attack_path,
        attack_domains = args.domains,

        # Vocab arguments
        vocab_path = args.vocab,
        vocab_freq = args.frequency,

        # Contextualizer arguments
        window = args.window,

        # Embedder arguments
        embedder      = args.embedder,
        dim_embedding = args.dim_embedding,
        dim_hidden    = args.dim_hidden,
        max_context   = args.max_context,
        train_ratio   = args.train_ratio,
        embedder_path = args.embedder_path,

        # Synonym finder arguments
        metric              = args.metric,
        n                   = args.n,
        threshold           = args.threshold,
        pos                 = not args.no_pos,
        synonym_finder_path = args.synonym_finder_path,
    )

    # Show pipeline
    summary(nlp)

    # Write nlp to directory
    nlp.to_disk(args.outdir)


if __name__ == '__main__':
    main()