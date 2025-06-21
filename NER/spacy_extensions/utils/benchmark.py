"""Utilities to measure the performance of each component in a pipeline."""
import argformat
import argparse
import coreferee
import spacy
import timeit
from collections import OrderedDict
from pathlib import Path
from spacy.util import get_model_meta, dict_to_dot, load_config, load_model_from_config
from spacy.language import Language
from spacy_extensions.pipeline import *
from time import time
from typing import Dict, Union


def benchmark_load(model_path: Union[str, Path], number: int=100):
    """Measure time to load from path."""
    # Check whether local model is given
    if not isinstance(model_path, str) or not Path(model_path).exists():
        raise ValueError("Could not find model at path '{model_path}'")

    # Get model path
    if not isinstance(model_path, Path):
        model_path = Path(model_path)

    # Initialize timings
    times = OrderedDict({'__iterations__': number})
    
    # Time meta
    times['meta'] = timeit.timeit(
        lambda: get_model_meta(model_path),
        number = number,
    )
    meta = get_model_meta(model_path)
    config_path = model_path / "config.cfg"

    # Time overrides
    config = dict()
    times['overrides'] = timeit.timeit(
        lambda: dict_to_dot(config),
        number = number,
    )
    overrides = dict_to_dot(config)

    # Time config
    times['config'] = timeit.timeit(
        lambda: load_config(config_path, overrides=overrides),
        number = number,
    )
    config = load_config(config_path, overrides=overrides)

    # Time model init
    start = time()
    nlp = load_model_from_config(
        config,
        vocab=True,
        meta=meta,
    )
    times['load_model_from_config'] = time() - start

    # Time all components
    for name, proc in nlp._components:
        if hasattr(proc, 'from_disk'):
            times[name] = timeit.timeit(
                lambda: proc.from_disk(model_path / name, exclude=['vocab']),
                number = number,
            )

    # Return report
    return times

    # Load individual pipes
    return nlp.from_disk(model_path, exclude=exclude, overrides=overrides)



def benchmark_processing(nlp:Language, text:str, number:int=100) -> Dict[str,float]:
    """Measure processing time for components of a pipeline for a given text.
    
        Parameters
        ----------
        nlp : spacy.language.Language
            Pipeline to time.
            
        text : str
            Example text to time when passing through pipeline.
        
        number : int, default=100
            Number of times to execute given pipe.
        """
    # Time ensure doc (tokenization)
    times = OrderedDict({
        '__iterations__': number,
        'ensure_doc': timeit.timeit(
            lambda: nlp._ensure_doc(text),
            number = number,
    )})
    
    # Get actual document
    doc = nlp._ensure_doc(text)
    
    # Loop over pipes
    for name, proc in nlp.pipeline:
        # Time given pipe
        times[name] = timeit.timeit(lambda: proc(doc),number=number)
        # Update doc
        doc = proc(doc)

    # Return times
    return times


def benchmark_report(times: OrderedDict) -> str:
    """Create a time report for the entire pipeline"""
    # Extract iterations
    number = 'N/A'
    if '__iterations__' in times:
        number = times['__iterations__']
        del times['__iterations__']

    # Compute total time
    total = sum(times.values())
    times['Total'] = total

    # Include ratio
    for key, value in times.items():
        times[key] = (value, value / total)

    # Get setup
    width = max(len(x)+1 for x in times.keys())

    # Create report
    report = f"\nPipeline time evaluation [n={number}]:\n"
    for pipe, (time, fraction) in times.items():
        # Convert time to different units
        if isinstance(number, int): time /= number
        unit = 's '
        if time < 0.001:
            time *= 1_000_000
            unit  = 'μs'
        elif time < 1:
            time *= 1000
            unit  = 'ms'

        report += f"  {pipe+':':{width}} {time:8.4f}{unit:2} ({fraction:7.2%})\n"

    # Return report
    return report


def parse_args():
    """Parse arguments from command line."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description     = 'description',
        formatter_class = argformat.StructuredFormatter,
    )

    # Add arguments
    parser.add_argument('nlp', help='pipeline to benchmark')
    parser.add_argument('file', help='file to use for benchmark')
    parser.add_argument('--n', type=int, default=1,
        help='number of runs for benchmark')

    # Parse arguments and return
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()

    # Benchmark loading
    print(benchmark_report(benchmark_load(args.nlp, args.n)))
    # Benchmark files
    nlp = spacy.load(args.nlp)
    with open(args.file) as infile:
        data = infile.read()
    print(benchmark_report(benchmark_processing(nlp, data, args.n)))


if __name__ == '__main__':
    main()
