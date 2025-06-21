# [markdown]
# # Test "Supervised" BERT Models

import torch
import loader
from const import ExpDataset

import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from train_common import TAU
from collections import defaultdict
from test_common import calc_results_per_document, calc_results_per_sentence


def get_innermost_dirs(base_dir):
    innermost_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if not dirs:
            rel_path = os.path.relpath(root, base_dir)
            innermost_dirs.append(rel_path)
    return [os.path.join(base_dir, dir) for dir in innermost_dirs]


# [markdown]
# ## Test functions
# The following snippet contains all the functions that, given a model and a set of (or a single) data loaders, will return the labels and predictions.
def run_test_multi_label_per_document(model, data_loaders):
    model.to(device)
    model.eval()

    docs = {}

    for doc_name, data_loader in tqdm(data_loaders.items()):
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch in data_loader:

                input_ids = torch.as_tensor(batch["input_ids"]).to(device)
                attn_mask = torch.as_tensor(batch["attention_mask"]).to(device)
                labels = torch.as_tensor(batch["labels"]).to(device)
                all_labels.extend(labels.cpu().numpy())
                out = model(input_ids=input_ids, attention_mask=attn_mask)
                probs = out.logits.sigmoid()
                preds = torch.where(probs > TAU, 1.0, 0.0)
                all_preds.extend(preds.cpu().numpy())

        all_labels = np.clip(np.sum(all_labels, axis=0, dtype=int), 0, 1).reshape(1, -1)
        all_preds = np.clip(np.sum(all_preds, axis=0, dtype=int), 0, 1).reshape(1, -1)

        docs[doc_name] = {"labels": all_labels, "preds": all_preds}

    return docs


def run_test_multi_label_per_sentence(model, data_loader):
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader):

            input_ids = torch.as_tensor(batch["input_ids"]).to(device)
            attn_mask = torch.as_tensor(batch["attention_mask"]).to(device)
            labels = torch.as_tensor(batch["labels"]).to(device)
            all_labels.extend(labels.cpu().numpy())
            out = model(input_ids=input_ids, attention_mask=attn_mask)
            probs = out.logits.sigmoid()
            preds = torch.where(probs > TAU, 1.0, 0.0)
            all_preds.extend(preds.cpu().numpy())

    docs = {"labels": all_labels, "preds": all_preds}

    return docs


def run_test_single_label_per_sentence(model, data_loader):
    raise NotImplementedError


def run_test_single_label_per_document(model, data_loaders):
    raise NotImplementedError


# [markdown]
# ## Test on single experimental setup
# Each setup is saved inside a folder named `"fine_tuned/{problem_type}/{conf\_id}\_{dataset}\_{model\_name}"`.
# Select the folder by its name and run the tests!


def run_test(model_dir, model_name, dataset_name, per_document=True):
    model, tokenizer = loader.load_finetuned_model(model_dir)
    _, _, data_loaders = loader.load_datasets(
        dataset_name, 16, tokenizer, per_document=per_document
    )
    is_single_label = None
    if (
        dataset_name == ExpDataset.BOSCH_TECHNIQUES_SL.value
        or dataset_name == ExpDataset.TRAM_TECHNIQUES_SL.value
    ):
        is_single_label = True
    else:
        is_single_label = False

    test_f = None
    if per_document and is_single_label:
        test_f = run_test_single_label_per_document
    elif per_document and not is_single_label:
        test_f = run_test_multi_label_per_document
    elif not per_document and is_single_label:
        test_f = run_test_single_label_per_sentence
    else:
        test_f = run_test_multi_label_per_sentence

    print("Setup:")
    print(
        f"\tdataset_name={dataset_name}\n\tmodel_name={model_name}\n\tis_single_label={is_single_label}\n\tper_document={per_document}"
    )
    print(f"\ttest_function={test_f}")

    results = test_f(model, data_loaders)

    # data loader contains the dataset labels as they are presented to the model when training
    # we can either load them from the constants, or just retrieve from the data loader
    # target_names = data_loaders[list(data_loaders.keys())[0]].dataset.labels

    if per_document:
        out_df = calc_results_per_document(results, model.config.id2label)
    else:
        out_df = calc_results_per_sentence(results)

    return out_df


def retrieve_setup_from_folder_name(folder):
    import os, json

    single_label = "single_label" in folder
    dataset = "_".join(os.path.basename(folder).split("_")[1:3])

    if "_".join(os.path.basename(folder).split("_")[1:4]) == "tram_ood_rebalanced":
        dataset = "tram_ood_rebalanced"
    # why didn't I just make better folder names
    elif dataset[:4] == "tram" and dataset not in [
        "tram",
        "tram_10",
        "tram_25",
        "tram_sl",
        "tram_artificial",
        "tram_ood",
        "tram_ood_rebalanced",
    ]:
        dataset = "tram"

    try:
        with open(os.path.join(folder, "model_params.json"), "r") as f:
            params = json.load(f)

        batch_size = params.get("batch_size")
        freeze_layers = params.get("freeze_layers")
        learning_rate = params.get("learning_rate")
        # if pos_weight is missing, it's 25. the experiments were run with a previous version of the
        # code that didn't contain the pos_weight parameter, but it was generated inside the function.
        end_factor = params.get("end_factor")
        pos_weight = params.get("pos_weight")

        with open(os.path.join(folder, "config.json"), "r") as f:
            model_name = json.load(f)["_name_or_path"]

        return {
            "model_name": model_name,
            "dataset": dataset,
            "batch_size": batch_size,
            "freeze_layers": freeze_layers,
            "learning_rate": learning_rate,
            "pos_weight": pos_weight,
            "end_factor": end_factor,
            "is_single_label": single_label,
        }
    except:
        print("missing config for: %s" % folder)


# device selection: you can choose gpu 0 with cuda:0 and gpu 1 with cuda:1
# device = "cuda:1" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="test_labeled",
        description="This code tests supervised BERT models stored inside a folder on their corresponding datasets.",
    )

    parser.add_argument(
        "model_dir",
        type=str,
        help='The base directory containing the models to test (e.g. "fine_tuned/tram_swipe").',
    )
    parser.add_argument(
        "outfile", type=str, help="The output file to save the results."
    )
    parser.add_argument(
        "device",
        type=str,
        help="The device to use for testing (e.g., 'cuda:0' or 'cpu').",
        default="cuda:0",
    )
    parser.add_argument(
        "--show-baseline",
        action="store_true",
        help="This flag will show the baseline results for the TRAM dataset.",
    )
    parser.add_argument(
        "--remove-dupl-models",
        action="store_true",
        help="This flag will remove duplicate models from the results (e.g., two RoBERTa trained with different hyper-parameters)",
    )

    args = parser.parse_args()

    device = args.device
    model_dir = args.model_dir
    outfile = args.outfile
    remove_dupl_models = args.remove_dupl_models

    selected_models = get_innermost_dirs(model_dir)

    setups = {}
    for f in selected_models:
        setups[f] = retrieve_setup_from_folder_name(f)

    # [markdown]
    # ### Collect test results per document
    final_results = {
        "model": [],
        "dataset": [],
        "freeze_layers": [],
        "batch_size": [],
        "learning_rate": [],
        "pos_weight": [],
        "end_factor": [],
        "accuracy_mean": [],
        "precision_mean": [],
        "recall_mean": [],
        "f1_mean": [],
    }

    for i, s in enumerate(setups):
        print("Evaluating setup %d/%d: %s" % (i + 1, len(setups), s))
        try:
            model_dir = s
            model_name = setups[s]["model_name"]
            dataset_name = setups[s]["dataset"]
            freeze_layers = setups[s]["freeze_layers"]
            pos_weight = setups[s]["pos_weight"]
            end_factor = setups[s]["end_factor"]
            learning_rate = setups[s]["learning_rate"]
            batch_size = setups[s]["batch_size"]
            result_df = run_test(model_dir, model_name, dataset_name)

            f1_mean = result_df["f1"].mean()
            accuracy_mean = result_df["accuracy"].mean()
            precision_mean = result_df["precision"].mean()
            recall_mean = result_df["recall"].mean()

            final_results["model"].append(model_name)
            final_results["dataset"].append(dataset_name)
            final_results["freeze_layers"].append(freeze_layers)
            final_results["pos_weight"].append(pos_weight)
            final_results["end_factor"].append(end_factor)
            final_results["learning_rate"].append(learning_rate)
            final_results["batch_size"].append(batch_size)
            final_results["f1_mean"].append(f1_mean)
            final_results["accuracy_mean"].append(accuracy_mean)
            final_results["precision_mean"].append(precision_mean)
            final_results["recall_mean"].append(recall_mean)
            print(f"{model_name}: {f1_mean}")
        except:
            print("\033[91m [!] Skipping %s ! \033[0m" % s)

    with pd.option_context("display.max_rows", None):
        final_df = pd.DataFrame(final_results)
        final_df = final_df.sort_values(by=["dataset", "model", "f1_mean"])
        final_df["f1_mean"] = (final_df["f1_mean"] * 100).round(2)
        final_df["precision_mean"] = (final_df["precision_mean"] * 100).round(2)
        final_df["recall_mean"] = (final_df["recall_mean"] * 100).round(2)
        if remove_dupl_models:
            out_table = final_df.loc[
                final_df.groupby("model")["f1_mean"].idxmax()
            ].reset_index(drop=True)[
                ["model", "f1_mean", "precision_mean", "recall_mean"]
            ]
        else:
            out_table = final_df[["model", "f1_mean", "precision_mean", "recall_mean"]]
        print(out_table)
        final_df.to_csv(outfile)

    # [markdown]
    # ## Test Baseline
    if args.show_baseline:
        model_name = "baseline_tram"
        model, tokenizer = loader.load_untrained_model(
            "scibert_multi_label_model", "tram"
        )
        _, _, data_loaders = loader.load_datasets(
            "tram", 16, tokenizer, per_document=True
        )
        results = run_test_multi_label_per_document(model, data_loaders)
        out_df = calc_results_per_document(results, model.config.id2label)
        print(f"Baseline results for {model_name} on TRAM dataset:")
        print(
            f"TRAM(original), f1_mean: {(out_df.f1.mean()*100).round(2)}, precision_mean: {(out_df.precision.mean()*100).round(2)}, recall_mean: {(out_df.recall.mean()*100).round(2)}"
        )
