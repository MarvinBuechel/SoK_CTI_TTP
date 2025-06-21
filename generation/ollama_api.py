import os
import ollama as ol
import numpy as np


OLLAMA_SERVER = os.environ["OLLAMA_API_URL"]

ollama_instance = None

# Connect to ollama server based on provided env variable OLLAMA_API_URL.
def ollama_init():
    global ollama_instance
    ollama_instance = ol.Client(host=OLLAMA_SERVER)


# Retrieve Embeddings from ollama server.
def get_embedding(text: str, model: str = "rjmalagon/gte-qwen2-7b-instruct:f16"):
    while True:
        try:
            emb_array = np.array(ollama_instance.embeddings(model=model, prompt=text)["embedding"])
            return emb_array
        except Exception as e:
            ollama_init()
            print("reconnect...")