#!/bin/bash

echo "Running documents through pipelines..."

# Run pipelines - TRAM
## Full pipeline
python3 -m spacy_extensions run \
    pipeline/full \
    ../data/TRAM/PlainText/* \
    output/TRAM/full/
    
## No POS pipeline
python3 -m spacy_extensions run \
    pipeline/pos \
    ../data/TRAM/PlainText/* \
    output/TRAM/pos/
    
## No lemmatization pipeline
python3 -m spacy_extensions run \
    pipeline/lemma \
    ../data/TRAM/PlainText/* \
    output/TRAM/lemma/

## No related words pipeline
python3 -m spacy_extensions run \
    pipeline/related_words \
    ../data/TRAM/PlainText/* \
    output/TRAM/related_words/

## No parsing pipeline
python3 -m spacy_extensions run \
    pipeline/parser_exact \
    ../data/TRAM/PlainText/* \
    output/TRAM/parser_exact/

## Naive
python3 -m spacy_extensions run \
    pipeline/naive \
    ../data/TRAM/PlainText/* \
    output/TRAM/naive/

## Very naive
python3 -m spacy_extensions run \
    pipeline/very_naive \
    ../data/TRAM/PlainText/* \
    output/TRAM/very_naive/




# Run pipelines - Bosch
## Full pipeline
python3 -m spacy_extensions run \
    pipeline/full \
    ../data/top_k/BoschAnnoCTR/PlainText/test/* \
    output/Bosch/full/
    
## No POS pipeline
python3 -m spacy_extensions run \
    pipeline/pos \
    ../data/top_k/BoschAnnoCTR/PlainText/test/* \
    output/Bosch/pos/

## No lemmatization pipeline
python3 -m spacy_extensions run \
    pipeline/lemma \
    ../data/top_k/BoschAnnoCTR/PlainText/test/* \
    output/Bosch/lemma/

## No related words pipeline
python3 -m spacy_extensions run \
    pipeline/related_words \
    ../data/top_k/BoschAnnoCTR/PlainText/test/* \
    output/Bosch/related_words/

## No parsing pipeline
python3 -m spacy_extensions run \
    pipeline/parser_exact \
    ../data/top_k/BoschAnnoCTR/PlainText/test/* \
    output/Bosch/parser_exact/

## Naive
python3 -m spacy_extensions run \
    pipeline/naive \
    ../data/top_k/BoschAnnoCTR/PlainText/test/* \
    output/Bosch/naive/

## Very naive
python3 -m spacy_extensions run \
    pipeline/very_naive \
    ../data/top_k/BoschAnnoCTR/PlainText/test/* \
    output/Bosch/very_naive/

echo "Documents completed!"