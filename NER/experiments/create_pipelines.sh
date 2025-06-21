#!/bin/bash

# Create output directory
mkdir -p pipeline
echo "Creating pipelines..."

# Create pipelines
## Full pipeline
python3 -m spacy_extensions pipeline \
    pipeline/full \
    --base en_core_web_trf \
    --tokenizer \
    --ioc \
    --pos ../models/pos.json \
    --lemma ../models/lemma.json \
    --wordnet ../models/synset.csv \
    --cti ../cti \
    --domains enterprise \
    --phrase ../models/phrases.json \
    --matcher subphrase

## No POS pipeline
python3 -m spacy_extensions pipeline \
    pipeline/pos \
    --base en_core_web_trf \
    --tokenizer \
    --ioc \
    --lemma ../models/lemma.json \
    --wordnet ../models/synset.csv \
    --cti ../cti \
    --domains enterprise \
    --phrase ../models/phrases.json \
    --matcher subphrase

## No lemmatization pipeline
python3 -m spacy_extensions pipeline \
    pipeline/lemma \
    --base en_core_web_trf \
    --tokenizer \
    --ioc \
    --pos ../models/pos.json \
    --wordnet ../models/synset.csv \
    --cti ../cti \
    --domains enterprise \
    --phrase ../models/phrases.json \
    --matcher subphrase

## No related words pipeline
python3 -m spacy_extensions pipeline \
    pipeline/related_words \
    --base en_core_web_trf \
    --tokenizer \
    --ioc \
    --pos ../models/pos.json \
    --lemma ../models/lemma.json \
    --cti ../cti \
    --domains enterprise \
    --phrase ../models/phrases.json \
    --matcher subphrase

## No parsing pipeline
python3 -m spacy_extensions pipeline \
    pipeline/parser_exact \
    --base en_core_web_trf \
    --tokenizer \
    --ioc \
    --pos ../models/pos.json \
    --lemma ../models/lemma.json \
    --wordnet ../models/synset.csv \
    --cti ../cti \
    --domains enterprise \
    --phrase ../models/phrases.json \
    --matcher exact

## Naive
python3 -m spacy_extensions pipeline \
    pipeline/naive \
    --base en_core_web_trf \
    --tokenizer \
    --ioc \
    --cti ../cti \
    --domains enterprise \
    --phrase ../models/phrases.json \
    --matcher exact

## Very naive
python3 -m spacy_extensions pipeline \
    pipeline/very_naive \
    --base en_core_web_trf \
    --cti ../cti \
    --domains enterprise \
    --phrase ../models/phrases.json \
    --matcher exact

echo "Pipelines created!"