# [markdown]
# # Test BERT-models for sentence similarity

#
import pickle
import loader
from const import MODEL_SENTENCE_SIM
from tqdm import tqdm
import torch
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import json
from test_common import calc_results_per_document
from const import (
    TRAM_TECHNIQUES_10_LABELS,
    TRAM_TECHNIQUES_25_LABELS,
    TRAM_TECHNIQUES_LABELS,
    BOSCH_TECHNIQUES_LABELS,
    BOSCH_TECHNIQUES_10_LABELS,
    BOSCH_TECHNIQUES_25_LABELS,
    BOSCH_TECHNIQUES_50_LABELS,
)
import argparse


def load_dataset(dataset_name, label_set):
    # load dataset. this time, we get a pandas DataFrame, as we don't need
    # to do any training. we just need to get the embeddings out of the
    # sentences

    LABEL_MAP = {
        "bosch_t": BOSCH_TECHNIQUES_LABELS,
        "bosch_t10": BOSCH_TECHNIQUES_10_LABELS,
        "bosch_t25": BOSCH_TECHNIQUES_25_LABELS,
        "bosch_t50": BOSCH_TECHNIQUES_50_LABELS,
        "tram": TRAM_TECHNIQUES_LABELS,
        "tram_10": TRAM_TECHNIQUES_10_LABELS,
        "tram_25": TRAM_TECHNIQUES_25_LABELS,
        "bosch+tram": sorted(
            list(set(BOSCH_TECHNIQUES_LABELS).union(set(TRAM_TECHNIQUES_LABELS)))
        ),
        "all_mitre": sorted(list(store.keys())),
    }

    if dataset_name == "bosch":
        df = loader.load_datasets_for_tuning_embedding_threshold("bosch_t")
    elif dataset_name == "tram":
        df = loader.load_datasets_for_tuning_embedding_threshold("tram")
    elif dataset_name == "bosch+tram":
        df_bosch = loader.load_datasets_for_tuning_embedding_threshold("bosch_t")
        df_tram = loader.load_datasets_for_tuning_embedding_threshold("tram")
        df = pd.concat([df_bosch, df_tram])
        df.reset_index(drop=True, inplace=True)
    else:
        raise Exception

    labels = LABEL_MAP[label_set]
    return df, labels



#
def find_hyperparams(model, model_name, df, ttps):
    store_loc = {k: v for k, v in store.items() if k in ttps}
    output = {}
    # they are already sorted, but who knows, sort them again
    # as for some reason the multilabelbinarizer decides to sort them on
    # its own. thus if you convert back to ttps and they do not follow
    # the same order, all the labels are wrong :D
    # ttps = sorted(list(store.keys()))
    lb = MultiLabelBinarizer()
    lb.fit([ttps])

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        output[index] = {}
        output[index]["labels"] = list(set(row.labels).intersection(ttps))

        sentence_emb = model.encode(row.sentence)
        ttp_embs = np.array(
            [store_loc[ttp][model_name]["emb"] for ttp in store_loc]
        ).reshape(-1, sentence_emb.shape[0])

        sim = model.similarity([sentence_emb], ttp_embs)
        output[index]["sim"] = sim

    best_f1 = -1
    best_tau = 0.5

    print("Finding best tau...")
    for tau in np.linspace(0.25, 0.75, num=20):
        all_labels = []
        all_preds = []

        for k in list(output.keys()):
            labels = output[k]["labels"]
            ttp_sims = output[k]["sim"]
            preds = torch.nonzero(ttp_sims[0] > tau).view(-1).tolist()
            preds = [list(store_loc.keys())[index] for index in preds]
            all_labels.append(labels)
            all_preds.append(preds)

        y_true = lb.transform(all_labels)
        y_pred = lb.transform(all_preds)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau

    return best_f1, best_tau



def test_model_per_document(model, model_name, df, dataset_name, ttps, tau):
    store_loc = {k: v for k, v in store.items() if k in ttps}

    if "bosch" in dataset_name:
        col = "document"
    elif "tram" in dataset_name:
        col = "doc_title"
    else:
        raise AttributeError

    out = {}
    for doc_name, df_doc in tqdm(df.groupby(col)):
        all_labels = []
        all_preds = []

        for _, row in df_doc.iterrows():

            labels = list(set(row.labels).intersection(ttps))

            sentence_emb = model.encode(row.sentence)
            ttp_embs = np.array(
                [store_loc[ttp][model_name]["emb"] for ttp in store_loc]
            ).reshape(-1, sentence_emb.shape[0])

            ttp_sim = model.similarity(sentence_emb, ttp_embs)

            preds = torch.nonzero(ttp_sim[0] > tau).view(-1).tolist()
            preds = [list(store_loc.keys())[index] for index in preds]
            all_labels.extend(labels)
            all_preds.extend(preds)

        out[doc_name] = {"labels": list(set(all_labels)), "preds": list(set(all_preds))}

    return out


def load_dataset_test(dataset_name, label_set):
    # load dataset. this time, we get a pandas DataFrame, as we don't need
    # to do any training. we just need to get the embeddings out of the
    # sentences

    LABEL_MAP = {
        "bosch_t": BOSCH_TECHNIQUES_LABELS,
        "bosch_t10": BOSCH_TECHNIQUES_10_LABELS,
        "bosch_t25": BOSCH_TECHNIQUES_25_LABELS,
        "bosch_t50": BOSCH_TECHNIQUES_50_LABELS,
        "tram": TRAM_TECHNIQUES_LABELS,
        "tram_10": TRAM_TECHNIQUES_10_LABELS,
        "tram_25": TRAM_TECHNIQUES_25_LABELS,
        "bosch+tram": sorted(
            list(set(BOSCH_TECHNIQUES_LABELS).union(set(TRAM_TECHNIQUES_LABELS)))
        ),
        "all_mitre": sorted(list(store.keys())),
    }

    if dataset_name == "bosch":
        df_test = loader.load_datasets_for_testing_embedding_threshold("bosch_t")
    elif dataset_name == "tram":
        df_test = loader.load_datasets_for_testing_embedding_threshold("tram")
    elif dataset_name == "bosch+tram":
        df_bosch = loader.load_datasets_for_testing_embedding_threshold("bosch_t")
        df_tram = loader.load_datasets_for_testing_embedding_threshold("tram")
        df_test = pd.concat([df_bosch, df_tram])
        df_test.reset_index(drop=True, inplace=True)
    else:
        raise Exception

    labels = LABEL_MAP[label_set]
    return df_test, labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tune hyper-parameters and test unlabeled classification models.")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Select mode: 'tuning' or 'test'.")
    
    # Training mode
    train_parser = subparsers.add_parser("tuning", help="Run hyperparameter tuning on decision threshold (tau)")
    train_parser.add_argument("tuning_output", type=str, help="Path to the output configuration json file (e.g., 'configs/thresholds.json')")
    train_parser.add_argument(
        "device",
        type=str,
        help="The device to use for tuning (e.g., 'cuda', 'cpu', 'mps' for apple silicon).",
        default="cuda",
    )

    # Testing mode
    test_parser = subparsers.add_parser("test", help="Test models")
    test_parser.add_argument("test_config", type=str, help="Path to the configuration file")
    test_parser.add_argument("test_output", type=str, help="Path to the output CSV file")
    test_parser.add_argument(
        "device",
        type=str,
        help="The device to use for testing (e.g., 'cuda', 'cpu', 'mps' for apple silicon).",
        default="cuda",
    )
    test_parser.add_argument("--open-set", action="store_true", help="If set, the model will be tested in open-set mode. Overrides test_config to 'configs/open_classification.json'.")

    args = parser.parse_args()

    device = args.device
    mode = args.mode

    # [markdown]
    # ## Load configuration files

    # load the precalculated embeddings.
    with open("datasets/mitre_embeddings.pickle", "rb") as f:
        store = pickle.load(f)

    model_names_all = MODEL_SENTENCE_SIM
    models = []
    model_names = []
    for mdl in model_names_all:
        try:
            models.append(loader.load_model_for_embedding(mdl).to(device))
            model_names.append(mdl)
        except:
            print("Failed loading %s" % mdl)

    model_params = {}

    if mode == "tuning":
        output_file = args.tuning_output

        label_sets = {
            "tram": ["tram", "all_mitre"],
            "bosch": ["bosch_t", "all_mitre"],
        }
        for dataset_name in ["tram", "bosch"]:
            model_params[dataset_name] = {}

            for label_set in label_sets[dataset_name]:

                df, labels = load_dataset(dataset_name, label_set)
                model_params[dataset_name][label_set] = {}

                # for each row in the dataset:
                # 1) calculate the embeddings for each model
                # 2) calculate the similarity to each encoded TTP title and TTP description
                #    making sure to consider the one calculated by the corresponding model.
                #    This can be easily done, as we pre-generated the vectors and saved them
                #    using the model name as the key
                for model, model_name in zip(models, model_names):
                    print(f"> {model_name}, {dataset_name}, {label_set}")

                    best_f1, best_tau = find_hyperparams(model, model_name, df, labels)
                    model_params[dataset_name][label_set][model_name] = {"tau": best_tau}

                    print(f"best_f1={best_f1}, best_tau={best_tau}\n")

        with open(output_file, "w") as f:
            json.dump(model_params, f)

    else:
        # [markdown]
        # ## Test
        config_file = args.test_config
        output_file = args.test_output

        if args.open_set:
            config_file = "configs/open_classification.json"

        # load json config file
        with open(config_file, "r") as f:
            config = json.load(f)

        output = {
            "model_name": [],
            "dataset_name": [],
            "label_set": [],
            "f1": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
        }

        for dataset in config:
            for label_set in config[dataset]:
                df_test, labels = load_dataset_test(dataset, label_set)
                for model_name in config[dataset][label_set]:
                    model = loader.load_model_for_embedding(model_name)
                    print(f"> testing {model_name}")
                    tau = config[dataset][label_set][model_name]["tau"]
                    model = loader.load_model_for_embedding(model_name)
                    out_df = test_model_per_document(
                        model, model_name, df_test, dataset, labels, tau
                    )
                    results_df = calc_results_per_document(out_df)
                    f1 = results_df.f1.mean()
                    accuracy = results_df.accuracy.mean()
                    precision = results_df.precision.mean()
                    recall = results_df.recall.mean()
                    print(f1)
                    output["model_name"].append(model_name)
                    output["dataset_name"].append(dataset)
                    output["label_set"].append(label_set)
                    output["f1"].append(f1)
                    output["accuracy"].append(accuracy)
                    output["precision"].append(precision)
                    output["recall"].append(recall)

        final_df = pd.DataFrame(output)
        final_df = final_df.sort_values(by=["dataset_name", "label_set", "model_name"])
        final_df["f1"] = (final_df["f1"] * 100).round(2)
        final_df["precision"] = (final_df["precision"] * 100).round(2)
        final_df["recall"] = (final_df["recall"] * 100).round(2)
        final_df = final_df[["model_name", "dataset_name", "label_set", "f1", "precision", "recall"]]

        if args.open_set:
            final_df = final_df.loc[
                final_df.groupby("label_set")["f1"].idxmax()
            ].reset_index(drop=True)[
                ["model_name", "dataset_name", "label_set", "f1", "precision", "recall"]
            ]

        print(final_df)
        final_df.to_csv(output_file)
