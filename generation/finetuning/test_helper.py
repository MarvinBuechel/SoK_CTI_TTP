import ast
import csv
import os

import numpy as np
import pandas as pd
import torch
from transformers import TextStreamer

finetuning_strategies = {
    "Name": str,
    "PP": bool,
    "RAG": bool,
    "FSP": bool,
    "JudgeAgent": bool,
    "DatasetTRAM": bool,
    "DatasetBosch": bool,
    "DocumentLevel": bool,
}



def get_dataset(path):
    df = pd.read_json(path)

    reports = df["cti_report"]
    mitre_ids = df["mitre_ids"]

    return zip(reports, mitre_ids)


def get_dataset_sentence(path):
    df = pd.read_json(path)

    reports = df["cti_report"]
    label = df["label"]

    return zip(reports, label)


def initialize_log_file(model, strategies: dict, file_path="training_log.csv"):
    """
    Initializes the CSV log file with hyperparameters from the model.

    Args:
        file_path (str): Path to the CSV file where logs will be saved.
        model (PreTrainedModel): Hugging Face model loaded with from_pretrained.
        :param strategies: (dict): Dictionary of finetuning strategies and their boolean values.
    """

    # Extract hyperparameters from the model's config
    hyperparameters = vars(model.config) if hasattr(model, "config") else {}

    # Write the header and hyperparameters ### if the file does not exist
    with open(file_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header for predictions and labels
        #header = ["Prediction", "Labels"]
        header = list(strategies.keys()) + list(hyperparameters.keys())
        writer.writerow(header)

        # Write an initial row with empty prediction/labels and hyperparameter values
        row = [strategies.get(key, "N/A") for key in strategies.keys()] + \
              [hyperparameters.get(key, "N/A") for key in hyperparameters.keys()]
        writer.writerow(row)


def log_single_prediction(prediction, labels, file_path="training_log.csv"):
    """
    Logs a single prediction and multiple labels to the CSV file.

    Args:
        prediction (list): List of The predicted value from the model.
        labels (list): List of ground truth labels for the prediction.
        file_path (str): Path to the CSV file where logs will be saved.
    """
    # Combine labels into a single string
    prediction_str = " ".join(prediction)
    labels_str = " ".join(labels)

    # Append prediction and labels to the file
    with open(file_path, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([prediction_str, labels_str])


def inference(model, tokenizer, user_message):
    text_streamer = TextStreamer(tokenizer)

    template_text = tokenizer.apply_chat_template(user_message, tokenize=False, add_generation_prompt=True)
    encoded_text = tokenizer.encode(template_text, add_special_tokens=False)

    with torch.inference_mode():
        inputs = {"input_ids": torch.tensor([encoded_text]).to("cuda")}
        output_ids = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=128,
            do_sample=False,
            top_p=1,
            repetition_penalty=1.2,
            temperature=0,
            num_return_sequences=1,
        )
        output_ids = output_ids.to("cpu")

        return tokenizer.decode(output_ids[0]).split("assistant<|end_header_id|>")[1]


def create_subsequences(document: str, n: int = 13, stride: int = 5) -> list[str]:
    #words = document.split()
    #subsequences = [' '.join(words[i:i+n]) for i in range(0, len(words), stride)]
    document = document.replace("\n", " ")
    subsequences = document.split(". ")
    return subsequences


def read_rag_db(path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="ISO-8859-1")
    df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    return df