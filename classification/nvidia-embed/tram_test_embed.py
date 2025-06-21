from transformers import AutoModel
from torch.nn import DataParallel
import loader
import pandas as pd

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import pickle


def load_dataset(dataset_name):
    # load dataset. this time, we get a pandas DataFrame, as we don't need
    # to do any training. we just need to get the embeddings out of the
    # sentences

    if dataset_name == "bosch":
        df = loader.load_datasets_for_testing_embedding_threshold("bosch_t")
    elif dataset_name == "tram":
        df = loader.load_datasets_for_testing_embedding_threshold("tram")
    else:
        raise Exception
    
    return df


class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        sentence = self.data.iloc[index].sentence
        index = self.data.index[index]
        return index, sentence
    
    def __len__(self):
        return self.len
    
model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True)
for module_key, module in model._modules.items():
    model._modules[module_key] = DataParallel(module)
model.to("cuda")


max_length = 4096
batch = 2

df = load_dataset("tram")
tram = TextDataset(df)
tram_loader = DataLoader(tram, batch_size=batch, shuffle=False)

all_embeddings = {}
for idx, sentences in tqdm(tram_loader):
    model.eval()
    sentences = list(sentences)
    embeddings = model.encode(sentences, instruction="", max_length=max_length).to("cpu")
    for id, emb in zip(idx, embeddings):
        all_embeddings[id] = emb


with open("nvidia-tram-test-embeddings.pickle", "wb") as f:
    pickle.dump(all_embeddings, f)
