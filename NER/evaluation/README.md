# Manual Evaluation
Here we provide scripts to compare our pipeline predictions with manually evaluated files.

During this evaluation, we assume:
 * `path/to/output` contains pipeline-processed documents (see `python3 -m spacy_extensions run -h`).
 * `path/to/manual` contains manually processed documents (using SpaCy-Annotator).
These files can be stored in different directories, but if that is the case, please replace the paths mentioned above with your custom paths during the evaluation.

## Run evaluation
Simply run the evaluation script as:

```bash
python3 evaluate.py path/to/output path/to/manual
```

**NB:** Make sure that all files in the `path/to/manual` directory have a corresponding file (i.e., same filename + extension) in the `path/to/output` directory.