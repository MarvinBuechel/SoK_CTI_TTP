# Named Entity Recognition
This directory contains the code of the Named Entity Recognition experiments in the *"SoK: Automated TTP Extraction from CTI Reports – Are We There Yet?"* paper. 

## Reproducing results

The most straightforward way of reproducing the named entity recognition experiments is via the provided Dockerfile. This Dockerfile installs the required packages (see [Installation](#installation)) and then runs the experiments (see [Usage](#usage)).
```bash
docker build . -t ner
docker run ner
```

> [!IMPORTANT]  
> The documents from both TRAM 2 and the Bosch AnnoCTR dataset are available in the `data/` directory. While these are the same documents as in the other experiments, their format is different as the NER pipelines take full raw text documents, rather than JSON format.

Besides using the Dockerfile, it is possible to manually reproduce the results. To this end, please follow the [Installation](#installation) instructions below. After, installing the required packages, you can run the `experiments/ablation.sh` file to reproduce the results from Table 5. This experiment performs the following actions:
 1. Create the NER pipelines required for each part of the ablation study. This runs the script `experiments/create_pipelines.sh` which executes the "pipeline generation" part of the [Usage](#usage) instructions for this experiment.
 2. Run the documents (see `data/` directory) through the created pipelines. This runs the script `experiments/run_pipelines.sh` which takes all TRAM2 and Bosch AnnoCTR documents and runs it through the created pipelines. The resulting parsed documents can be found in the `output/` directory that is created when running the script.
 3. Evaluate the parsed documents by comparing the named entities with the ground truth labels. This runs the script `experiments/evaluate_pipelines.sh` which takes all parsed documents and their ground truth labels to perform the evaluation. Results of this evaluation are stored in the `results/`
 directory that is created when running the script.
 
## Installation
Currently, `spacy-extensions` can only be installed by locally downloading the repository.
Next, you can install the module using `pip`:

```
pip3 install -e /path/to/directory/containing/setup.py/
```

### Dependencies
Python dependencies can easily be installed from pip using the `requirements.txt` file:

```
pip3 install -r requirements.txt
```

Besides the python dependencies, we require the following components to be installed:

#### SpaCy pipelines
```bash
python3 -m spacy download en_core_web_sm
python3 -m spacy download en_core_web_lg
python3 -m spacy download en_core_web_trf
```

#### NLTK modules
```bash
python3 -m nltk.downloader wordnet
```

#### CTI - MITRE ATT&CK framework
The MITRE ATT&CK framework can be downloaded as follows:

```bash
git clone https://github.com/mitre/cti.git
cd cti
git checkout ATT&CK-v14.1
```

Note that we have tested everyting on the MITRE ATT&CK framework version 14.1, hence we need to checkout that release tag.

## Usage
We provide a command-line utility for using spacy-extensions to process CTI reports.

```bash
python3 -m spacy_extensions pipeline <args>
python3 -m spacy_extensions run <args>
python3 -m spacy_extensions evaluate <args>
```

### Pipeline generation
First we need to configure and create an NLP pipeline that we want to use for processing.
To this end, we run spacy_extensions in `pipeline` mode:

```bash
python3 -m spacy_extensions pipeline \
  path/to/pipeline/outdir \
  --base en_core_web_trf \
  --tokenizer \
  --ioc \
  --pos models/pos.json \
  --lemma models/lemma.json \
  --wordnet models/synset.csv \
  --coref \
  --cti path/to/mitre/cti \
  --domains enterprise \
  --phrases models/phrases.json \
  --matcher subphrase
```

For our default configuration please use the following parameters:
 - `--base en_core_web_trf`, the base pipeline. Make sure that this is downloaded using `python3 -m spacy download en_core_web_trf`
 - `--tokenizer`, adds tokenizer exceptions to deal with references in text.
 - `--ioc`, ensures indicator of compromise (IoC) detection.
 - `--pos models/pos.json`, to ensure correct POS for MITRE ATT&CK framework.
 - `--lemma models/lemma.json`, to ensure correct lemmas for MITRE ATT&CK framework.
 - `--wordnet models/synset.csv`, to allow related word detection of MITRE ATT&CK framework.
 - `--coref`, enables coreference resolution.
 - `--cti pat/to/mitre/cti`, to load MITRE ATT&CK framework. See [dependencies](#cti---mitre-attck-framework) how to download cti.
 - `--domains enterprise`, to only load enterprise part of ATT&CK framework.
 - `--phrases models/phrases.json`, to load additional subphrases (variants of cti).
 - `--matcher subphrase`, enables subphrase detection (recommended). Other options are:
   - `subphrase`, detects directly linked subphrases in parse tree (recommended).
   - `sentence`, detects all subphrases if they occur in a sentence.
   - `exact`, only matches exact order of subphrase.

### Run pipeline
To run the pipeline for given documents, simply run:

```bash
python3 -m spacy_extensions run \
  /path/to/pipeline \
  /path/to/input/*.txt \
  /path/to/output/dir
```

Here we give:
 - `nlp`, path to the pipeline. Should be same as `path/to/pipeline/outdir` in pipeline generation phase.
 - `inputs`, path(s) to input `txt` files.
 - `output`, path to output directory in with processed files will be stored.

### Evaluate parsed documents
Once we have processed all documents, we can perform an evaluation:

```bash
python3 -m spacy_extensions evalute \
  /path/to/processed/files/*.json \
  /path/to/labels.json \
  --filter 'T[0-9]{4}' \
  --force-technique
```

Here we give:
 - `input`, path to input files. Generally this should be set to `/path/to/output/dir/*` from run pipeline.
 - `labels`, path to labels file (see below).
 - `--filter <regex>`, filter to apply when processing files. If set, we will only show evaluation of labels that follow regex. E.g., `'T[0-9]{4}'` will only show MITRE ATT&CK techniques.
 - `--force-technique`, if set, it will rewrite subtechniques to their parent technique: `Txxxx.yyy -> Txxxx`.

#### labels.json
To perform the evaluation, we should have a `labels.json` file that includes the labels for each processed document. This should have the following form:

```
{
  "processed_doc_1.json": ["label1", "label2"],
  "processed_doc_2.json": ["T0000", "T0001", "T0002"],
  "processed_doc_3.json": ["T0000"]
}
```
