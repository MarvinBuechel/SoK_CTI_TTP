from transformers import AutoModel
from torch.nn import DataParallel
import loader
import pandas as pd

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
import pickle


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

df = df = loader.load_datasets_for_tuning_embedding_threshold("bosch")
bosch = TextDataset(df)
bosch_loader = DataLoader(bosch, batch_size=batch, shuffle=False)

all_embeddings = {}
for idx, sentences in tqdm(bosch_loader):
    model.eval()
    sentences = list(sentences)
    embeddings = model.encode(sentences, instruction="", max_length=max_length).to("cpu")
    for id, emb in zip(idx, embeddings):
        all_embeddings[id] = emb


with open("nvidia-bosch-train-embeddings.pickle", "wb") as f:
    pickle.dump(all_embeddings, f)
