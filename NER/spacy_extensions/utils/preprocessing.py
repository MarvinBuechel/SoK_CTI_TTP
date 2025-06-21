from spacy.tokens import Doc, DocBin
from spacy.vocab  import Vocab
from typing import Iterable
from tqdm import tqdm

def load_preprocessed(
        vocab: Vocab,
        paths: Iterable[str],
        verbose: bool = False,
    ) -> Iterable[Doc]:
    """Load preprocessed files into Doc objects.
    
        Parameters
        ----------
        vocab : Vocab
            Vocabulary to use for loading the documents.
            Often, you can use ``nlp.vocab`` for the given pipeline that you
            will run.
        
        paths : Iterable[str]
            Paths from which to load preprocessed documents.

        verbose : bool, default=False
            If True, print progress
        
        Returns
        -------
        documents : Iterable[Doc]
            Preprocessed documents loaded from paths.
        """
    # Initialise docbin
    doc_bin = DocBin()

    # Add verbosity
    if verbose: paths = tqdm(paths, desc="Merging")

    # Merge with docbins loaded from paths
    for path in paths:
        doc_bin.merge(DocBin().from_disk(path))

    # Get iterator
    iterator = doc_bin.get_docs(vocab)

    # Add verbosity
    if verbose: iterator = tqdm(
        iterator,
        desc  = "Recreating Docs",
        total = len(doc_bin),
    )

    # Return documents
    return iterator


def load_files_txt(files: Iterable[str]) -> Iterable[str]:
    """Load text files from disk.
        
        Parameters
        ----------
        files : Iterable[str]
            Paths to files to load as text.
            
        Yields
        ------
        text : str
            Text loaded from files.
        """
    for file in files:
        with open(file) as infile:
            yield infile.read()


def load_files_docbin(files, vocab):
    """Load preprocessed DocBin files from disk. See ``load_preprocessed``"""
    return list(load_preprocessed(vocab, files, verbose=True))