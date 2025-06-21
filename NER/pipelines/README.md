# Pipelines
This directory contains a `__main__.py` script for building pipelines from command-line.
Furthermore, we offer various pre-trained pipes.
We support the following base pipes:
 * `cti_base_sm`
 * `cti_base_trf`

We support the following full `sm` pipes:
 * `cti_bert_sm`
 * `cti_elmo_sm`
 * `cti_nlm_sm`
 * `cti_word2vec_cbow_sm`
 * `cti_word2vec_skipgram_sm`

We support the following full `trf` pipes:
 * `cti_bert_trf`
 * `cti_elmo_trf`
 * `cti_nlm_trf`
 * `cti_word2vec_cbow_trf`
 * `cti_word2vec_skipgram_trf`

## Building pipes
We build the pipes using the `__main__.py` script.
Note that this `__main__` script can be executed as `pipelines` from the parent directory, which is the convention that we use here.
Note that we use the following paths:
 * `/path/to/mitre/cti_v10.1/`: Cloned from `git@github.com:mitre/cti.git`, ATT&CK-v10.1 branch.
 * `/path/to/vocab/{base}`: Vocab generated using `create_vocab.py` script.

### Base pipes
We create base pipes based on MITRE ATT&CK CTI v10.1 and both the SpaCy `en_core_web_sm` and `en_core_web_trf` base pipelines.
These base pipes can be used to detect IOCs, ATT&CK concepts and create a vocabulary used for embedders.

#### cti_base_sm
```bash
python3 pipelines cti_base_sm\
    --base en_core_web_sm\
    --cti /path/to/mitre/cti_v10.1/
```

#### cti_base_trf
```bash
python3 pipelines cti_base_trf\
    --base en_core_web_trf\
    --cti /path/to/mitre/cti_v10.1/
```

### Embedder pipes
Using the vocab obtained by the base pipes, we created pretrained embedder pipes for both the SpaCy `en_core_web_sm` and `en_core_web_trf` base pipelines.

#### cti_word2vec_cbow_sm
```bash
python3 pipelines cti_word2vec_cbow_sm\
    --base en_core_web_sm
    --cti /path/to/mitre/cti_v10.1/\
    --vocab /path/to/vocab/cti_base_sm\
    --embedder word2vec-cbow
    --embedder-path /path/to/pretrained/embedder.model
    --synonym-finder-path /path/to/pretrained/synonym_finder.sf
```
