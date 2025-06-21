from openai import OpenAI
import os
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

import spacy
import tiktoken 
import numpy as np
import pickle 
import tqdm


client = OpenAI()
_nlp = spacy.load("en_core_web_sm")


SEED = 0xCAFEBABE
GEN_MODEL = "gpt-3.5-turbo-1106"
EMB_MODEL = "text-embedding-ada-002"
TEMPERATURE = 0
MAX_TOKEN = 4096
TAU = 0.80


def __get_response(prompt):
    global client
    response = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKEN,
        seed=SEED,
        messages=prompt,
    )
    # fingerprint = response.system_fingerprint
    return response.choices[0].message.content


def gen_embeddings(text: str) -> list:
    """Generate embeddings for the given text."""
    global client
    response = client.embeddings.create(
        model=EMB_MODEL,
        input=text,
    )
    return response.data[0].embedding


def summarization(text: str) -> str:
    # Original prompt of Siracusano et al. (2023)
    """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:
    """
    return __get_response([
        {
            "role": "system",
            "content": "Write a concise summary of the following:",
        },
        {"role": "user", "content": text},
        {"role": "assistant", "content": "CONCISE SUMMARY:"},
    ])


def extract_attack_patterns_1(text: str) -> str:
    return __get_response([
        {
            "role": "system",
            "content": "Use the following portion of a long document to "
                    "see if any of the text is relevant to answer the "
                    "question. Return any relevant text verbatim.\n",
        },
        {"role": "user", "content": text},
        {"role": "user", "content": "Question: Which techniques are used by the attacker?\n Report only Relevant text, if any"},
        {"role": "assistant", "content": "RELEVANT TEXT:"},
    ])


def extract_attack_patterns_2(text: str) -> str:
    return __get_response([
        {
            "role": "system",
            "content": "Describe step by step the key facts in the following text:",
        },
        {"role": "user", "content": text},
        {"role": "assistant", "content": "KEY FACTS:"},
    ])


def gen_embedding_from_mitre_store():
    # note 
    # output must be {technique_id: embedding}
    mitre_df = pd.read_csv("mitre_table.csv", encoding="ISO-8859-1")
    picle_data = {}

    import re
    TECH_ID_RE = re.compile(r"\bT\d+(?:\.\d+)?\b")

    for ind in mitre_df.index:
        print(f"Processing {ind} of {len(mitre_df)}")
        name = mitre_df["name"][ind]
        description = mitre_df["description"][ind]
        id = mitre_df["ID"][ind]
        if not bool(TECH_ID_RE.match(id)):
            print("Skipping invalid technique ID:", id)
        text = f"{name}\n{description}"
        embedding = gen_embeddings(text)
        picle_data[id] = embedding

        with open("mitre_store.pkl", "wb") as f:
            pickle.dump(picle_data, f)

def cosine_similarity(u, v) -> float:
    u, v = np.array(u), np.array(v)
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
              # shift to [0, 1]


def split_sentences(text: str) -> str:
    doc = _nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def match_tecniques(sentence: str) -> list:
    sent_emb = gen_embeddings(sentence)
    with open("mitre_store.pkl", "rb") as f:
        store = pickle.load(f)
    matches = []
    for tech_id, emb in store.items():
        sim = cosine_similarity(sent_emb, emb)
        if sim >= TAU:
            matches.append(tech_id)
    return matches


def chunk_text(text: str,
               max_tokens: int = MAX_TOKEN,
               model: str = GEN_MODEL) -> list[str]:

    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)           # List[int]
    chunks: list[str] = []
    
    for start in range(0, len(tokens), max_tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))

    return chunks


def main(text: str) -> list:
    ttps = []
    print("Chunking text...")
    chunks = chunk_text(text, max_tokens=MAX_TOKEN, model=GEN_MODEL)
    print("Chunks created:", len(chunks))

    summaries = []
    for chunk in chunks:
        summaries.append(summarization(chunk))
    
    print("Summaries created:", len(summaries))
    summary = "\n".join(summaries)
    print(summary)

    
    sentences = []
    relevant_text = extract_attack_patterns_1(summary)
    print("Relevant text extracted:", relevant_text)
    key_facts = extract_attack_patterns_2(relevant_text)
    print("Key facts extracted:", key_facts)
    sentences.extend(split_sentences(key_facts))
    print("Sentences extracted:", len(sentences))
    
    for s in sentences:
        print(s)
        out = match_tecniques(s)
        print(out)
        ttps.extend(out)

    return list(set(ttps))


if __name__ == "__main__":
    with open("input.txt", "r") as f:
        text = f.read()
    print(main(text))
    # gen_embedding_from_mitre_store()