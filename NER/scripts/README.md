# Scripts for SpaCy-Extensions
This directory contains various scripts for the `spacy_extensions` library.

## Preprocessing
When working with large corpora, one often needs to run pipelines multiple times, e.g., when training and testing embedders.
In such instances, it is prefered to preprocess the documents in the corpus as to reduce the time for running the pipeline over and over again.
We recommend preprocessing documents with one of the `cti_base_*` pipes.
Afterwards, when loading a pipe extending said base, you can use `spacy_extensions.utils.pipeline.nlp2preprocessed()` to transform the pipeline into a pipe that assumes documents have already been preprocessed.

### Example
```python
# Imports
import spacy
from spacy_extensions.utils.pipeline import nlp2preprocessed

base_pipe = spacy.load('cti_base_sm')
```

## Vocab
To use embedders, we require a vocabulary for which to train the embedders.
Such a vocabulary can be created from one of the base pipelines.
To do so, we run the `vocab.py` script, which takes as input the `pipeline` used to process documents, the `path/to/vocab` and the `files` from which to create the vocab.

### Example
```
python3 vocab.py base_cti_sm sm.vocab /paths/to/input/files/*
```

## Training embedders
When using an embedder in the pipeline, it must be trained. Therefore, we provide the `embedder.py` script to train an embedder using a pipeline that contains an untrained embedder. See `pipelines` README.md for how to create a pipeline with an untrained embedder.

```
python3 embedder.py\
    path/to/pipeline\       # The pipeline containing an untrained embedder
    embedder.model\         # Output file to store embedder
    /path/to/input/files/*\ # Input files to train with
    --preprocessed\         # Optional, if set, assumes input files have been preprocessed
    --epochs 100\           # Number of epochs to train with, defaults to 100 epochs
    --batch-size 128        # Batch size to train with, defaults to 128
```

## Training synonym finder
