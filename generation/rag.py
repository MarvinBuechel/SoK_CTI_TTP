import ast
from typing import Any
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

import ollama_api


def create_rag_db(db_df: pd.DataFrame, text_culumn_name="text", text_culumn_name2=None, rag_db_name="rag_db.csv"):
    db_df['embedding'] = None

    for ind in tqdm(db_df.index):
        text = db_df[text_culumn_name][ind]
        if text_culumn_name2 is not None:
            text += " - " + db_df[text_culumn_name2][ind]

        embedding = ollama_api.get_embedding(text)
        embedding = list(embedding)
        db_df.at[ind, "embedding"] = embedding

    db_df.to_csv(rag_db_name)


def read_rag_db(path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    return df



def get_best_text_embeddings(text: str, embeddings, n_best: int = 100):
    text_embedding = ollama_api.get_embedding(text)

    while text_embedding is None:
        text_embedding = ollama_api.get_embedding(text)

    return get_n_closest_vector(text_embedding, embeddings, n_best)


def euk_dis(a, b):
    return np.linalg.norm(a - b)


def cos_dis(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_n_closest_vector(s_vec, embeddings, n: int) -> list[dict[str, int]]:
    distance_list = []
    for i, vec in enumerate(embeddings):
        distance_list.append({"index": i, "distance": euk_dis(a=s_vec, b=vec)})

    sorted_list = sorted(distance_list, key=lambda x: x["distance"])

    return sorted_list[:n]


def get_paragraph_embeddings(text: str, embeddings, n_per_paragraph):
    text_parts = list(filter(lambda x: x != '', re.split(r"\n{2,}", text)))

    source_embeddings = [ollama_api.get_embedding(part) for part in text_parts]

    embeddings_per_paragraph = []
    for s_vec in source_embeddings:
        embeddings_per_paragraph.extend(get_n_closest_vector(s_vec, embeddings, n_per_paragraph))

    sorted_list = sorted(embeddings_per_paragraph, key=lambda x: x["distance"])
    embeddings_per_paragraph = remove_duplicates_by_index(
        sorted_list)  # remove after sort to avoid deleting the closest distance

    return embeddings_per_paragraph


def split_text(string, chunk_size, overlap):
    """
    Splits the input string into chunks of size `chunk_size` with overlap of `overlap` characters.

    :param string: input string to be chunked
    :param chunk_size: size of the chunks
    :param overlap: number of overlapping characters between chunks
    :return: list of string chunks
    """
    chunks = []

    for i in range(0, len(string) - overlap, chunk_size - overlap):
        chunks.append(string[i:i + chunk_size])

    return chunks


def get_chunk_embeddings(text: str, embeddings, n_per_chunk, chunk_size=500, overlap=20):
    text_parts = split_text(text, chunk_size, overlap)

    source_embeddings = [ollama_api.get_embedding(part) for part in text_parts]

    embeddings_per_chunk = []
    for s_vec in source_embeddings:
        embeddings_per_chunk.extend(get_n_closest_vector(s_vec, embeddings, n_per_chunk))

    sorted_list = sorted(embeddings_per_chunk, key=lambda x: x["distance"])
    embeddings_per_chunk = remove_duplicates_by_index(
        sorted_list)  # remove after sort to avoid deleting the closest distance

    return embeddings_per_chunk


def reverse_embedding(answer_text: str, embeddings, n_per_line: int = 1):
    text_parts = list(filter(lambda x: x != '', re.split(r"\n{1,}", answer_text)))

    line_embeddings = [ollama_api.get_embedding(part) for part in text_parts]

    best_embeddings = []
    for s_vec in line_embeddings:
        best_embeddings.extend(get_n_closest_vector(s_vec, embeddings, n_per_line))

    return best_embeddings


def remove_duplicates_by_index(lst):
    unique_indices = set()
    result = []
    for item in lst:
        if 'index' in item:
            index = item['index']
            if index not in unique_indices:
                unique_indices.add(index)
                result.append(item)
        else:
            result.append(item)
    return result


def shrink_distance(lst, shrink_factor=0.5):
    lst = sorted(lst, key=lambda x: x["distance"])

    # Create a dictionary to store index as key and current shrunk distance as value
    dic = {}
    # Loop through the list of dictionaries
    for i in lst:
        if i['index'] not in dic:
            dic[i['index']] = i['distance']
        else:
            # Shrink the distance by the specified factor
            dic[i['index']] *= shrink_factor

    # Create a new list of dictionaries with shrunk distances
    new_lst = [{'index': k, 'distance': v} for k, v in dic.items()]

    new_lst = sorted(new_lst, key=lambda x: x["distance"])

    return new_lst


def average_distance(lst):
    lst = sorted(lst, key=lambda x: x["distance"])

    # Create a dictionary to store index as key and all related distances as value
    dic = {}

    # Loop through the list of dictionaries
    for i in lst:
        if i['index'] not in dic:
            # If the index is not in the dic yet, create a new list with the distance as the first item
            dic[i['index']] = [i['distance']]
        else:
            # If the index is already in the dic, append the new distance to the related list
            dic[i['index']].append(i['distance'])

    # Create a new list of dictionary with averaged distances
    new_lst = [{'index': k, 'distance': sum(v) / len(v)} for k, v in dic.items()]

    new_lst = sorted(new_lst, key=lambda x: x["distance"])

    return new_lst


def get_best_list_embeddings(text_list: list[str], embeddings, n_best: int):
    distance_list = []

    for text in text_list:
        distance_list.extend(get_best_text_embeddings(text, embeddings, n_best))  # int(n_best/len(text_list))+1

    # distance_list = shrink_distance(distance_list, shrink_factor=1-(1/len(distance_list)))
    # distance_list = average_distance(distance_list)

    sorted_list = sorted(distance_list, key=lambda x: x["distance"])
    sorted_list = remove_duplicates_by_index(sorted_list)  # remove after sort to avoid deleting the closest distance

    return sorted_list[:n_best]


def df_rag_filter(df: pd.DataFrame, rag_df: pd.DataFrame, n_best: int, text_column_name: str = "text",
                  rag_embedding_column_name: str = "embedding"):
    if n_best == 0:
        return []

    rag_df = rag_df.reset_index(drop=True)
    best_embeddings = get_best_list_embeddings(df[text_column_name], rag_df[rag_embedding_column_name], n_best)
    best_ids = [item.get('index') for item in best_embeddings if 'index' in item]

    best_rag_rows = rag_df[rag_df.index.isin(best_ids)]
    return best_rag_rows


def str_df_rag_filter(text: str, rag_df: pd.DataFrame, n_best: int, rag_embedding_column_name: str = "embedding"):
    if n_best == 0:
        return []

    rag_df = rag_df.reset_index(drop=True)
    best_embeddings = get_best_text_embeddings(text, rag_df[rag_embedding_column_name], n_best)
    best_ids = [item.get('index') for item in best_embeddings if 'index' in item]

    best_rag_rows = rag_df[rag_df.index.isin(best_ids)]
    return best_rag_rows

