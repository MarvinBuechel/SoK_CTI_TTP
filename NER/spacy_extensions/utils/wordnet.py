from nltk.corpus import wordnet as wn

"""Mapping from wordnet POS to SpaCy POS."""
POS_WN_TO_SPACY = {
    wn.ADJ: 'ADJ',
    wn.ADV: 'ADV',
    wn.NOUN: 'NOUN',
    wn.VERB: 'VERB',
}

"""Mapping from SpaCy POS to wordnet POS."""
POS_SPACY_TO_WN = {
    'ADJ': wn.ADJ,
    'ADV': wn.ADV,
    'NOUN': wn.NOUN,
    'VERB': wn.VERB,
}