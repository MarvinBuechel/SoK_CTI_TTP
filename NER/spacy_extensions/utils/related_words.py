from pathlib import Path
import ndjson
import pandas as pd
from tqdm import tqdm
from typing import Dict, Iterable, Optional, Set

class RelatedWordsDF(pd.DataFrame):

    @classmethod
    def from_ndjson(
            cls,
            infile: str,
            sources: Optional[Iterable[str]] = {'w2v', 'reddit-slashes', 'ol', 'cn5', 'wn', 'wiki', 'swiki'},
            nrows  : Optional[int] = None,
            verbose: bool = False,
        ) -> pd.DataFrame:
        """Get RelatedWords from ndjson file.
        
            Parameters
            ----------
            infile : str
                Path to input ndjson file from which to read RelatedWords.

            sources : Optional[Iterable[str]], default={'w2v', 'reddit-slashes', 'ol', 'cn5', 'wn', 'wiki', 'swiki'},
                If given, use these as fixed sources. Set to None to recompute.
                By default, all sources are taken into account.

            nrows : Optional[int], default=None,
                Maximum number of rows to read from ndjson file.
                
            verbose : bool, default=False
                If True, prints progress.
            """
        # Get sources if not given
        if sources is None:
            sources = RelatedWords.get_sources(infile, nrows, verbose)

        # Set data
        data = {
            'word'   : list(),
            'related': list(),
            'score'  : list(),
        }
        for source in sources:
            data[f'source_{source}'] = list()
        
        # Open data
        with open(infile) as infile:
            # Read ndjson
            reader = ndjson.reader(infile)

            # Add verbosity if required
            if verbose: reader = tqdm(reader, desc="Loading ndjson ")

            # Get terms
            for index, entry in enumerate(reader):
                # Break if maximum number of rows is reached
                if nrows and index >= nrows: break

                # Unpack entry
                term = entry['word']
                for related in entry['related']:
                    word   = related['word']
                    score  = related['score']

                    # Add entry to data
                    data['word'   ].append(term)
                    data['related'].append(word)
                    data['score'  ].append(score)

                    # Set sources (False by default)
                    for key, value in data.items():
                        if key.startswith('source_'):
                            value.append(False)

                    # Set source to True if in dataset
                    for source in related['from']:
                        data[f'source_{source}'][-1] = True

        # Return as dataset
        return cls(data)


    @staticmethod
    def get_sources(
            infile : str,
            nrows  : Optional[int] = None,
            verbose: bool = False,
        ) -> Set[str]:
        """Get sources from input file.
        
            Parameters
            ----------
            infile : str
                Input ndjson file from which to get sources.

            nrows : Optional[int], default=None,
                Maximum number of rows to read from ndjson file.
                
            verbose : bool, default=False
                If True, prints progress.
            
            Returns
            -------
            sources : Set[str]
                Sources found in ndjson infile."""
        # Initialise result
        sources = set()

        # Open data
        with open(infile) as infile:
            # Read ndjson
            reader = ndjson.reader(infile)

            # Add verbosity if required
            if verbose: reader = tqdm(reader, desc="Loading sources")

            # Get sources for each entry
            for index, entry in enumerate(reader):
                # Break if maximum number of rows is reached
                if nrows and index >= nrows: break

                # Add related sources
                for related in entry['related']:
                    sources |= set(related['from'])

        # Return sources
        return sources

def ndjson2related(
        infile: Path,
        nrows: float = float('inf'),
        verbose: bool = False,
    ) -> Dict[str, dict]:
    """Read related words from ndjson file."""
    # Initialise result
    result = dict()

    # Open file
    with open(infile) as infile:
        # Get ndjson reader
        reader = ndjson.reader(infile)

        # Add verbosity if necessary
        if verbose: reader = tqdm(reader, desc='Loading ndjson')

        # Loop over all entries
        for index, entry in enumerate(reader):
            # Stop at max rows
            if index >= nrows: break

            # Save entry in result
            result[entry['word']] = entry

    # Return result
    return result