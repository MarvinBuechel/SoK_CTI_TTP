import os
import json
from scipy import spatial
from sentence_transformers import SentenceTransformer
import pandas as pd
import nltk

bert_model = SentenceTransformer('all-mpnet-base-v2')

df = pd.read_csv('enterprise-techniques.csv')

attack_pattern_dict = {} 
technique_mapping = {}

prev_id = None

for _, row in df.iterrows():
    _id = row['ID']
    if not pd.isnull(_id):
        attack_pattern_dict[_id] = [[row['Name'], row['Description']]]
        prev_id = _id
        technique_mapping[row['Name']] = _id
    else:
        attack_pattern_dict[prev_id].append([row['Name'], row['Description']])
        technique_mapping[row['Name']] = prev_id

embedding_memo = {}
def get_embedding(txt):
    if txt in embedding_memo:
        return embedding_memo[txt]
    emb = bert_model.encode([txt])[0]
    embedding_memo[txt] = emb
    return emb

def get_embedding_distance(txt1, txt2):
    p1 = get_embedding(txt1)
    p2 = get_embedding(txt2)
    score = spatial.distance.cosine(p1, p2)
    return score

def get_mitre_id(text):
    min_dist = 25
    ret = None
    for k, tech_list in attack_pattern_dict.items():
        for v in tech_list:
            # v[0] -> attack pattern title, v[1] -> description
            d = (0.5*get_embedding_distance(text, v[0]) + 0.5*get_embedding_distance(text, v[1]))
            if d < min_dist:
                min_dist = d
                ret = k
    return ret, min_dist

def remove_consec_newline(s):
    ret = s[0]
    for x in s[1:]:
        if not (x == ret[-1] and ret[-1]=='\n'):
            ret += x
    return ret

def get_all_attack_patterns(text, th=0.6):
    mapped = {}
    
    text = remove_consec_newline(text)
    text = text.replace('\t', ' ')
    text = text.replace("\'", "'")
    sents_nltk = nltk.sent_tokenize(text)
    sents = []
    for x in sents_nltk:
        sents += x.split('\n')
    for line in sents:
        if len(line) > 0:
            _id, dist = get_mitre_id(line)
            if dist < th:
                if _id not in mapped:
                    mapped[_id] = dist, line
                else:
                    if dist < mapped[_id][0]:
                        mapped[_id] = mapped[_id] = dist, line
    return mapped

ret = get_all_attack_patterns('test_output/litepower.txt', th=0.6)

for k, v in ret.items():
    print(k)
