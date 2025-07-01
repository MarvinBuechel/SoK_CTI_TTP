"""Pipeline tools for SpaCy extensions."""
# Ignore spacy warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Imports
import argformat
import argparse
import glob
import json
import pandas as pd
import re
import spacy
from collections import Counter, defaultdict, OrderedDict
from functools import partial
from itertools import cycle
from pathlib import Path
from spacy.matcher import Matcher
from spacy.tokens import Doc
from spacy_extensions.matcher import SentenceMatcher, SubphraseMatcher
from spacy_extensions.matcher.utils import (
    sentence2pattern,
    pattern_related,
    cti2phrases,
)
from spacy_extensions.pipeline import *
from spacy_extensions.utils.exceptions import (
    add_pos_exceptions,
    add_lemmatizer_exceptions,
)
from spacy_extensions.utils.iocs import IOCs
from tqdm import tqdm
from typing import (
    Any, Dict, Iterator, List, Literal, Optional, Set, Tuple, Union
)

################################################################################
#                               Operation modes                                #
################################################################################

def pipeline(args: argparse.Namespace) -> None:
    """Run in pipeline creation mode."""
    # Create pipeline
    print("Loading base...")
    nlp = spacy.load(args.base)

    # Add special tokenization cases
    if args.tokenizer:
        print("Adding tokenizer exceptions...")
        # Add tokenizer exceptions
        suffixes = nlp.Defaults.suffixes + [r"\[\d+\]"]
        suffixes = spacy.util.compile_suffix_regex(suffixes)
        nlp.tokenizer.suffix_search = suffixes.search

    # Add IoC pipe if required
    if args.ioc:
        print("Adding IoCs...")
        before = 'transformer' if args.base == 'en_core_web_trf' else 'tok2vec'
        ioc : MatcherIoC = nlp.add_pipe('matcher_ioc', before=before)
        ioc.add_regexes(IOCs)
        # Merge IoCs into single entity
        nlp.add_pipe("merge_entities", after='matcher_ioc')

    # Add POS exceptions if required
    if args.pos:
        print("Adding POS exceptions...")
        with open(args.pos) as infile:
            pos = json.load(infile)
        nlp = add_pos_exceptions(nlp, pos)
    else:
        nlp.disable_pipe('tagger')

    # Add lemma exceptions if required
    if args.lemma:
        print("Adding lemma exceptions...")
        with open(args.lemma) as infile:
            lemma = json.load(infile)
        nlp = add_lemmatizer_exceptions(nlp, lemma)

    # Add related words if required
    if args.wordnet or args.cti or args.phrases:
        print("Adding related words...")
        nlp.add_pipe('token_base')
        nlp.add_pipe('related')
        wordnet : RelatedWordNet = nlp.add_pipe('related_wordnet')

    # Add wordnet relations if required
    if args.wordnet:
        print("Adding wordnet-related words...")
        wordnet = wordnet.from_disk(args.wordnet)

    # Add cti matcher
    if args.cti or args.phrases:
        # Disable original NER
        nlp.disable_pipe('ner')
        # Add NER matcher
        print("Adding NER matcher...")
        ner_matcher : NERMatcher = nlp.add_pipe('ner_matcher')
        # Add matcher depending on given arguments
        if args.matcher == 'exact':
            matcher = Matcher(nlp.vocab)
        elif args.matcher == 'sentence':
            matcher = SentenceMatcher(nlp.vocab, '_.token_base' if args.lemma else 'LOWER')
        elif args.matcher == 'subphrase':
            matcher = SubphraseMatcher(nlp.vocab, '_.token_base' if args.lemma else 'LOWER')

        # Initialise phrases and labels
        phrases = list()
        labels  = list()

        # Add cti patterns
        if args.cti:
            phrases_, labels_ = zip(*cti2phrases(
                path = args.cti,
                domains = args.domains,
            ))
            phrases.extend(list(phrases_))
            labels .extend(list(labels_))

        # Add custom phrase patterns
        if args.phrases:
            with open(args.phrases) as infile:
                for label_, phrases_ in json.load(infile).items():
                    for phrase in phrases_:
                        phrases.append(phrase)
                        labels.append(label_)
            
        # Generate patterns
        phrases = nlp.pipe(phrases)
        # No lemma pattern
        if not args.lemma:
            get_pattern = partial(
                sentence2pattern,
                token_pattern=lambda x: {"LOWER": x.lower_},
            )
        else:
            get_pattern = partial(
                sentence2pattern,
                token_pattern=pattern_related,
            )
        # Exclude by POS only if we have POS entries
        if not args.pos: get_pattern = partial(get_pattern, exclude_pos=[])

        patterns = tqdm(
            map(get_pattern, phrases),
            desc = 'Generating patterns',
            total = len(labels),
        )
        for key, pattern in zip(labels, patterns):
            if len(pattern):
                matcher.add(key, [pattern])

        # Add matcher as NER matcher
        ner_matcher.add('cti', matcher)

    # Write pipeline to disk
    print("Writing to disk...")
    nlp.to_disk(args.output)
    print("Done!")


def run(args: argparse.Namespace) -> None:
    """Run pipeline to parse documents."""
    # Load nlp pipeline
    nlp = spacy.load(args.nlp)
    # Parse files
    files = map(read_file, args.input)
    docs = nlp.pipe(files, batch_size=16)

    # Write output
    outdir = Path(args.output)
    # Ensure output directory exists
    outdir.mkdir(parents=True, exist_ok=True)
    # Create to_json underscore
    underscore = ['ner_matches'] if Doc.has_extension('ner_matches') else None
    # Add iterator
    iterator = zip(args.input, docs)
    # Add progressbar
    iterator = tqdm(iterator, desc="Parsing documents", total=len(args.input))
    # Write each file to output
    for filename, doc in iterator:
        # Prepare name of outfile
        outfile = outdir / (Path(filename).stem + '.json')
        # Get document as json
        data = doc.to_json(underscore=underscore)
        # Write to disk
        with open(outfile, 'w') as o:
            json.dump(data, o)
            

def evaluate(args: argparse.Namespace) -> None:
    """Evaluate pipeline using given labels."""
    # Load predictions
    predictions = OrderedDict()
    for filename in args.input:
        with open(filename) as infile:
            # Load document
            doc = json.load(infile)
            # Get prediction
            if 'ner_matches' not in doc.get('_'):
                warnings.warn(
                    f"Doc did not contain 'ner_matches' defaulting to regular "
                    f"ents: '{filename}'."
                )
                prediction = doc['ents']
            else:
                prediction = doc['_']['ner_matches']
            # Set predictions
            predictions[filename] = [pred['label'] for pred in prediction]

    # Load labels
    with open(args.labels) as infile:
        labels = json.load(infile)

    # Rewrite to List
    y_pred = list()
    y_true = list()

    for filename, prediction in predictions.items():
        # Lookup filename label
        label = labels.get(filename)
        if label is None: label = labels.get(Path(filename).name)
        if label is None: label = labels.get(Path(filename).stem)
        # Check if entry is present as label
        if label is None:
            raise ValueError(f"Could not find entry for '{filename}'")
        # Add label to result
        y_pred.append(prediction)
        y_true.append(label)

    # Rewrite techniques if required
    if args.force_technique:
        y_pred = collapse_subtechnique(y_pred)
        y_true = collapse_subtechnique(y_true)

    # Print report
    return create_reports(
        y_true = y_true,
        y_pred = y_pred,
        files = args.input,
        filter = re.compile(args.filter) if args.filter else None
    )

################################################################################
#                              Auxiliary methods                               #
################################################################################

def read_file(filename: Union[str, Path]) -> str:
    """Read contents from file as text."""
    with open(filename) as infile:
        return infile.read()

def read_doc_labels(filename: Union[str, Path]) -> str:
    """Read doc as json."""
    with open(filename) as infile:
        doc = json.load(infile)
    print(doc.keys())

def collapse_subtechnique(labels: List[List[str]]) -> List[List[str]]:
    """Collapse subtechniques in the list of subtechniques.
    
        Parameters
        ----------
        labels : List[List[str]]
            Labels to rewrite.
            
        Returns
        -------
        labels : List[List[str]]
            Labels where subtechniques have been reduced to their parent
            technique.
        """
    # Initialise regex
    regex = re.compile('(T[0-9]{4}).[0-9]{3}')
    # Crete rewrite
    rewrite = lambda x: regex.match(x).group(1) if regex.match(x) else x
    # Map labels and return
    return [list(map(rewrite, entry)) for entry in labels]


def create_reports(
        y_true: List[List[str]],
        y_pred: List[List[str]],
        files: Optional[List[str]] = None,
        filter: Optional[re.Pattern] = None,
    ) -> None:
    """Perform evaluation with given true labels and prediction labels.

        Note
        ----
        Unfortunately, we cannot simply use the standard library
        sklearn.metrics.classification_report because we predict a set of labels
        per report. The classification_report only works with a fixed set of
        labels, per entry and as that is not yet known in advance, we cannot use
        that method. Therefore, we implement our own, based on set comparison.
    
        Parameters
        ----------
        y_true : List[List[str]]
            List of actual labels.
            
        y_pred : List[List[str]]
            List of predicted labels.

        files : Optional[List[str]]
            Optional filenames.

        filter : Optional[re.Pattern]
            Regex on which to match predictions and labels. You can use this to
            focus on predictions of a specific type.
        """
    # Perform checks
    if not len(y_true) == len(y_pred):
        raise ValueError(
            f"y_true and y_pred are not of same shape: {len(y_true)} !="
            f"{len(y_pred)}."
        )

    # Filter labels if required
    if filter:
        y_true = [[x for x in y if filter.match(x)] for y in y_true]
        y_pred = [[x for x in y if filter.match(x)] for y in y_pred]

    # Create reports
    print("Classification report unweighted:")
    print(create_report_per_file(y_true, y_pred, False, files))
    print()
    print("Classification report weighted:")
    print(create_report_per_file(y_true, y_pred, True, files))
    print()
    print("Classification labels unweighted:")
    print(create_report_per_label(y_true, y_pred, False))
    print()
    print("Classification labels weighted:")
    print(create_report_per_label(y_true, y_pred, True))
    print()
    print("Confusion report unweighted:")
    print(create_statistics_report(y_true, y_pred, False))
    print()
    print("Confusion report weighted:")
    print(create_statistics_report(y_true, y_pred, True))
    print()


def create_report_per_file(
        y_true: List[List[str]],
        y_pred: List[List[str]],
        weighted: bool = True,
        files: Optional[List[str]] = None,
    ) -> str:
    """Create a per-file report.
    
        Parameters
        ----------
        y_true : List[List[str]]
            List of actual labels.
            
        y_pred : List[List[str]]
            List of predicted labels.

        weighted : bool, default=True
            If True show weighted statistics.

        files : Optional[List[str]]
            Optional filenames.
        
        Returns
        -------
        report : str
            Generated report.
        """
    # Add files if not available
    if files is None:
        files = cycle(['N/A'])

    # Initialise report
    report = defaultdict(list)

    # Compute TP, FP, FN, weights per report
    metric = metrics(y_true, y_pred, weighted=weighted)
    for name, (TP, FP, FN, weights) in zip(files, metric):
        report['name'].append(name)
        report['TP'].append(sum(weights[x] for x in TP))
        report['FP'].append(sum(weights[x] for x in FP))
        report['FN'].append(len(FN))

    # Create report and return
    return data2report(report)


def create_report_per_label(
        y_true: List[List[str]],
        y_pred: List[List[str]],
        weighted: bool = True,
    ) -> str:
    """Create a per-label report.
    
        Parameters
        ----------
        y_true : List[List[str]]
            List of actual labels.
            
        y_pred : List[List[str]]
            List of predicted labels.

        weighted : bool, default=True
            If True show weighted statistics.
        
        Returns
        -------
        report : str
            Generated report.
        """
    # Initialise report: label -> metric (TP/FP/FN) -> int
    report = defaultdict(lambda: defaultdict(int))

    # Compute TP, FP, FN, weights per report
    for TP, FP, FN, weights in metrics(y_true, y_pred, weighted=weighted):
        # Add label performance
        for label in TP | FP | FN:
            # Update report data
            report[label]['TP'] += bool(label in TP) * weights.get(label, 1)
            report[label]['FP'] += bool(label in FP) * weights.get(label, 1)
            report[label]['FN'] += bool(label in FN)

    # Clean up report
    report_ = defaultdict(list)
    for key, entry in sorted(report.items()):
        report_['name'].append(key)
        report_['TP'].append(entry['TP'])
        report_['FP'].append(entry['FP'])
        report_['FN'].append(entry['FN'])

    # Return report
    return data2report(report_)


def create_statistics_report(
        y_true: List[List[str]],
        y_pred: List[List[str]],
        weighted: bool = True,
    ) -> str:
    """Create a report containing statistics of mispredictions.
    
        Parameters
        ----------
        y_true : List[List[str]]
            List of actual labels.
            
        y_pred : List[List[str]]
            List of predicted labels.

        weighted : bool, default=True
            If True show weighted statistics.
        
        Returns
        -------
        report : str
            Generated report.
        """
    # Initialise result
    TPs = Counter()
    FPs = Counter()
    FNs = Counter()
    
    # Compute TP, FP, FN, weights per report
    for TP, FP, FN, weights in metrics(y_true, y_pred, weighted=weighted):
        TPs .update({key: weights.get(key, 1) for key in TP})
        FPs.update({key: weights.get(key, 1) for key in FP})
        FNs.update({key: weights.get(key, 1) for key in FN})

    # Transform to report
    keys = list(sorted(set(TPs) | set(FPs) | set(FNs)))
    data = list()
    for key in keys:
        data.append({
            'name': key,
            'TP': TPs.get(key, 0),
            'FP': FPs.get(key, 0),
            'FN': FNs.get(key, 0),
            'support': TPs.get(key, 0) + FPs.get(key, 0) + FNs.get(key, 0),
        })
    return pd.DataFrame(data).to_string(index=False)


def data2report(
        data: Dict[Literal['name', 'TP', 'FP', 'FN'], List[int]],
        digits: int = 4,
    ) -> str:
    """Create classification report from given data.
    
        Parameters
        ----------
        data : Dict[Literal['name', 'TP', 'FP', 'FN'], List[int]]
            Data for which to create a report.
            
        digits : int, default=4
            Precision digits.

        Returns
        -------
        report : str
            String representation of report.
        """
    # Initialise result
    result = list()

    # Metrics
    precision = lambda TP, FP: TP / max(1, (TP+FP))
    recall    = lambda TP, FN: TP / max(1, (TP+FN))
    f1_score  = lambda TP, FP, FN: 2*TP / max(1, (2*TP + FP + FN))

    # Add individual reports
    for name, TP, FP, FN in zip(data['name'],data['TP'],data['FP'],data['FN']):
        result.append({
            'name'     : name,
            'precision': precision(TP, FP),
            'recall'   : recall   (TP, FN),
            'f1-score' : f1_score (TP, FP, FN),
            'support'  : TP + FP + FN,
        })
    
    # Compute total results
    TP = sum(data['TP'])
    FP = sum(data['FP'])
    FN = sum(data['FN'])
    result.append({
        'name'     : 'Average',
        'precision': precision(TP, FP),
        'recall'   : recall   (TP, FN),
        'f1-score' : f1_score (TP, FP, FN),
        'support'  : TP + FP + FN,
    })

    # Print as result
    result = pd.DataFrame(result)
    pd.options.display.precision = digits
    result.style.set_properties(**{
        'text-align': 'right',
    })
    return result.to_string(index=False)


def metrics(
        y_true: List[List[str]],
        y_pred: List[List[str]],
        weighted: bool = True,
    ) -> Iterator[Tuple[
        Set[str],
        Set[str],
        Set[str],
        Dict[str, int],
    ]]:
    """Compute TP, FP, FN and weight per report.
    
        Parameters
        ----------
        y_true : List[List[str]]
            List of actual labels.
            
        y_pred : List[List[str]]
            List of predicted labels.
            
        weighted : bool, default=True
            Whether to compute actual weights or simply compute 1 for each.

        Yields
        ------
        TP : Set[str]
            True positive samples for each document in y_true/y_pred.

        FP : Set[str]
            False positive samples for each document in y_true/y_pred.

        FN : Set[str]
            False negative samples for each document in y_true/y_pred.

        weights : Dict[str, int]
            Weights per sample for each document in y_true/y_pred.
        """
    # Loop over labels
    for y_true_, y_pred_ in zip(y_true, y_pred):

        # TN are not defined in our case
        TP = set(y_true_) & set(y_pred_) # TP are present in both pred in true
        FP = set(y_pred_) - set(y_true_) # FP were predicted, but shouldn't be
        FN = set(y_true_) - set(y_pred_) # FN were not predicted, but should be

        # Compute weights if required
        weights = Counter(y_pred_) if weighted else {x: 1 for x in y_pred_}

        # Yield result
        yield TP, FP, FN, weights

################################################################################
#                                     Main                                     #
################################################################################

def parse_args():
    """Parse arguments from command line."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description     = 'SpaCy extensions',
        formatter_class = argformat.StructuredFormatter,
    )
    # Create subparsers
    subparsers = parser.add_subparsers(
        dest = 'mode',
        help = 'mode of operation',
    )

    # Pipeline creation
    parser_pipeline = subparsers.add_parser(
        'pipeline', help='Create new NLP pipeline')
    parser_pipeline.add_argument('output', help='path to output dir')
    parser_pipeline.add_argument('--base', default='en_core_web_trf',
        help='existing pipeline to use as base')
    parser_pipeline.add_argument('--tokenizer' , action='store_true',
        help='if set, add tokenizer exceptions')
    parser_pipeline.add_argument('--ioc' , action='store_true',
        help='if set, include IoC detection')
    parser_pipeline.add_argument('--pos',     help='path to pos exceptions')
    parser_pipeline.add_argument('--lemma',   help='path to lemma exceptions')
    parser_pipeline.add_argument('--wordnet', help='path to wordnet relations')
    parser_pipeline.add_argument('--cti',     help='path to MITRE ATT&CK cti')
    parser_pipeline.add_argument('--domains', nargs='+',
        default=['enterprise'], help='path to MITRE ATT&CK domains')
    parser_pipeline.add_argument('--phrases', help='path to detection phrases')
    parser_pipeline.add_argument(
        '--matcher',
        choices = ('exact', 'sentence', 'subphrase'),
        default = 'subphrase',
        help = 'perform cti matching using given matcher',
    )


    # Run pipeline
    parser_run = subparsers.add_parser(
        'run', help='Run documents through pipeline')
    parser_run.add_argument('nlp', help='path to nlp pipeline to use')
    parser_run.add_argument('input', nargs='+', help='path to input files')
    parser_run.add_argument('output', help='path to output dir')


    # Evaluate parsed documents
    parser_evaluate = subparsers.add_parser(
        'evaluate', help='Evaluate pipeline performance'
    )
    parser_evaluate.add_argument(
        'input',
        nargs = '+',
        help = 'path to processed docs',
    )
    parser_evaluate.add_argument('labels', help='path to labels json file')
    parser_evaluate.add_argument('--filter', help='regex filter for labels')
    parser_evaluate.add_argument(
        '--force-technique',
        action = 'store_true',
        help = 'if set, rewrites subtechniques Txxxx.yyy -> Txxxx'
    )

    # Parse arguments
    args = parser.parse_args()

    # Perform file expansion if required
    if 'input' in args and len(args.input) == 1:
        args.input = glob.glob(args.input[0])

    # Return arguments
    return args


def main():
    # Parse arguments
    args = parse_args()

    # Run in given mode
    if args.mode == 'pipeline':
        return pipeline(args)
    elif args.mode == 'run':
        return run(args)
    elif args.mode == 'evaluate':
        return evaluate(args) 


if __name__ == '__main__':
    main()