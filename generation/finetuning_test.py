from itertools import chain
import numpy as np
from tqdm import tqdm

import mitre
import llm_response
import rag
import finetuning.cti_datasets.tram.tram_classes as tram
import finetuning.create_datasets.cti_preprocessing as cti_preprocessing
import finetuning.test_helper as th

import pandas as pd

TRAM_CLASSES = tram.TRAM_CLASSES


def test(
    strategies: th.finetuning_strategies,  # Dictionary defining evaluation strategy parameters
    model,  # Trained model for inference
    tokenizer,  # Corresponding tokenizer
    model_description: str = "model",  # Identifier for model logging
    test_len=0  # Optional limit on test dataset size
):
    """
    Evaluates a model's performance on TTP extraction tasks.

    Args:
        strategies (Dict): Strategy configuration dictionary containing flags like:
            - DatasetBosch/TRAM (dataset source)
            - DocumentLevel (granularity: full report vs sentences)
            - RAG (use retrieval-augmented generation)
            - FSP (few-shot prompting)
            - NumberOfLabel (filter specific labels)
            - JudgeAgent (use secondary model to refine predictions)
        model: Trained language model for inference
        tokenizer: Tokenizer for model input/output
        model_description (str): Name/identifier for the model (used in logging)
        test_len (int): Optional limit on number of test samples to process

    Returns:
        Tuple: Mean metrics across all test samples:
            - f1_score (float): Average F1 score
            - precision (float): Average precision
            - recall (float): Average recall
    """

    predictions = []  # Stores prediction results for each test sample
    dataset = []  # Test dataset

    # Load test dataset based on selected dataset
    if strategies["DatasetBosch"]:
        dataset.extend(th.get_dataset("finetuning/cti_datasets/bosch/bosch_cti_test_ds.json"))
    if strategies["DatasetTRAM"]:
        dataset.extend(th.get_dataset_sentence("finetuning/cti_datasets/tram/official_sok_split/test_split.json"))

    if test_len > 0: # Limit dataset size for testing
        dataset = dataset[:test_len]

    # Load precomputed RAG database (MITRE ATT&CK embeddings)
    mitre_table = th.read_rag_db("database/rag_db_qwen.csv")

    # Metric tracking variables
    f1_list = []
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list_direct = []
    acc_list_direct = []
    precision_list_direct = []
    recall_list_direct = []
    predicted_ids_direct = []
    predicted_ids = []

    # Logging configuration and initialize log file
    LOG_PATH = "./finetuning/experiments/" + model_description.split("/")[-1] + "_" + strategies["Name"]
    th.initialize_log_file(model=model, strategies=strategies, file_path=LOG_PATH + ".csv")

    # Process each test sample
    for data in tqdm(dataset):
        text_list, label = data # text_list: input text, label: ground truth MITRE IDs

        # Sentence-level vs Document-level processing
        if strategies["DocumentLevel"]:
            if strategies["DatasetBosch"]:
                sentences = [text_list]  # Whole CTI report as single sentence
            else:
                sentences = ["".join(text_list)] # Concatenate sentences for TRAM
        else:
            if strategies["DatasetBosch"]:
                sentences = th.create_subsequences(text_list) # Split report into sentences
            else:
                sentences = text_list # Use as-is for TRAM

        # Variable to accumulate responses for non-judge-agent runs
        text_responses = ""

        for i, sentence in enumerate(sentences):
            if sentence == "": # Skip empty sentences
                continue

            # Pre-Processing (cleaning) (not used in the paper)
            if strategies["PP"]:
                sentence = cti_preprocessing.replace_iocs(sentence)

            # Prompt generation logic
            if not strategies["DocumentLevel"]: # Sentence-level strategy
                if strategies["RAG"]: # RAG-based prompting
                    closest_embeddings = rag.get_best_text_embeddings(sentence, mitre_table["embedding"])
                    user_query = mitre.create_mitre_sentence_based_rag_prompt(sentence=sentence, closest_embeddings=closest_embeddings, mitre_table=mitre_table,
                                                                              few_shot = strategies["FSP"], is_bosch=strategies["DatasetBosch"],
                                                                              label_to_limit=strategies["NumberOfLabel"])
                else: # Baseline prompting
                    user_query = mitre.create_mitre_sentence_based_prompt(sentence, few_shot=strategies["FSP"], only_techniques=strategies["DatasetBosch"])
            else: # Document-level strategy
                if strategies["RAG"]: # RAG-based document-level prompting
                    closest_embeddings = rag.get_best_text_embeddings(sentence, mitre_table["embedding"])
                    user_query = mitre.create_mitre_based_rag_prompt(report=sentence, closest_embeddings=closest_embeddings, mitre_table=mitre_table, rag_entries=20,
                                                                              few_shot=strategies["FSP"], is_bosch=strategies["DatasetBosch"],
                                                                              label_to_limit=strategies["NumberOfLabel"])
                else: # Baseline prompting
                    user_query = mitre.create_mitre_based_prompt(sentence, few_shot=strategies["FSP"], only_techniques=strategies["DatasetBosch"])


            # Construct llm instruct message with system instruction and user query
            user_message = [
                {"role": "system",
                 "content": "You are a Cyber Security Expert who finds MITRE ATT&CK framework concepts in CTI Reports."},
                {"role": "user", "content": user_query}
            ]


            # Inference step: Get model response for the current sentence
            text_response = th.inference(model, tokenizer, user_message)


            # LLM Judge Agent: Second-stage validation of predictions (not used in the paper)
            if strategies["JudgeAgent"]:
                # Extract all valid MITRE concepts from model response
                predicted_concept_list = llm_response.getConceptInfos(mitre_table, text_response)
                predicted_ids = [predict_id[0] for predict_id in predicted_concept_list]
                if strategies["DatasetTRAM"]:  # remove all other techniques
                    predicted_concept_list = [candidate for candidate in predicted_concept_list if candidate[0] in tram.TRAM_CLASSES]
                if strategies["DatasetBosch"]:  # remove all other techniques
                    predicted_concept_list = [candidate for candidate in predicted_concept_list if "T1" in candidate[0] and "." not in candidate[0]]
                if len(predicted_concept_list) == 0:  # skip judge if candidate list is empty
                    continue
                judge_prompt = mitre.judge_sentence_based(sentence, predicted_concept_list)
                judge_message = [
                    {"role": "system",
                     "content": "You are a Cyber Security Expert and Judge who finds MITRE ATT&CK framework concepts in CTI Reports."},
                    {"role": "user", "content": judge_prompt}
                ]
                judge_response = th.inference(model, tokenizer, judge_message)
                judge_ids = llm_response.extract_mitre_ids(mitre_table, judge_response)
                intersection_ids = list(set(judge_ids) & set(predicted_ids))
                predicted_ids_direct += intersection_ids
            else:
                # If no judge agent, accumulate raw responses
                text_responses += text_response

        # Final prediction processing (if not using JudgeAgent)
        if not strategies["JudgeAgent"]:
            predicted_ids = llm_response.replaceNametoID(mitre_table, text_responses)
            predicted_ids_direct = llm_response.extract_mitre_ids(mitre_table, text_responses)

        # Dataset-specific filtering
        # Only use techniques and no sub-techniques
        if strategies["DatasetBosch"] and len(strategies["NumberOfLabel"]):
            label = [item for item in label if item in strategies["NumberOfLabel"]]
            predicted_ids = [item for item in predicted_ids if item in strategies["NumberOfLabel"]]
            predicted_ids_direct = [item for item in predicted_ids_direct if item in strategies["NumberOfLabel"]]

        # only use TRAM2 classes
        if strategies["DatasetTRAM"]:
            flattened_label = list(chain.from_iterable(label))  # flatten label list for document-wise evaluation
            label = list(set(flattened_label))
            if [] in label: # remove label of empty sentences in test dataset for evaluation
                label.remove([])

            predicted_ids = [item for item in predicted_ids if item in TRAM_CLASSES]
            predicted_ids_direct = [item for item in predicted_ids_direct if item in TRAM_CLASSES]

        # Remove duplicate predictions
        predicted_ids = list(set(predicted_ids))
        predicted_ids_direct = list(set(predicted_ids_direct))

        # Calculate evaluation metrics
        f1_score_name, precision_name, recall_name, acc_name = llm_response.calculate_metrics(predicted_ids, label)
        f1_score_id, precision_id, recall_id, acc_id = llm_response.calculate_metrics(predicted_ids_direct, label)

        # Store results for current test sample
        predictions.append(
            {
                "true_label": label,
                "predictions_name": predicted_ids,
                "predictions_id": predicted_ids_direct,
                "f1_score_name": f1_score_name,
                "acc_name": acc_name,
                "precision_name": precision_name,
                "recall_name": recall_name,
                "f1_score_id": f1_score_id,
                "acc_id": acc_id,
                "precision_id": precision_id,
                "recall_id": recall_id,
            })

        # Print metrics for current sample
        print("\n\nLabel IDs:")
        print(*list(set(label)))
        print("Found names: ")
        print(*predicted_ids)
        print("Found direct IDs: ")
        print(*predicted_ids_direct)

        print(fr"Name: F1: {f1_score_name:.2} - Recall: {recall_name:.2} - Precision: {precision_name:.2}")
        print(
            fr"ID:   F1: {f1_score_id:.2} - Recall: {recall_id:.2} - Precision: {precision_id:.2}")

        # Aggregate metrics for all samples
        f1_list.append(f1_score_name)
        acc_list.append(acc_name)
        precision_list.append(precision_name)
        recall_list.append(recall_name)
        f1_list_direct.append(f1_score_id)
        acc_list_direct.append(acc_id)
        precision_list_direct.append(precision_id)
        recall_list_direct.append(recall_id)

        # Print running average metrics
        print(":::::::::::::::::::::")
        print(
            fr"Name: F1: {np.mean(f1_list) :.3} - Recall: {np.mean(recall_list):.3} - Precision: {np.mean(precision_list):.3}")
        print(
            fr"ID:   F1: {np.mean(f1_list_direct):.3} - Recall: {np.mean(recall_list_direct):.3} - Precision: {np.mean(precision_list_direct):.3}")

    # Final metrics summary
    name_score = f"Name: F1: {np.mean(f1_list):.3} - Recall: {np.mean(recall_list):.3} - Precision: {np.mean(precision_list):.3}\n"
    id_score = f"ID:   F1: {np.mean(f1_list_direct):.3} - Recall: {np.mean(recall_list_direct):.3} - Precision: {np.mean(precision_list_direct):.3}\n"

    # Convert predictions to DataFrame for logging
    df_predictions = pd.DataFrame(predictions)

    # Log results to CSV file
    with open(LOG_PATH+".csv", "a") as logfile:
        df_predictions.to_csv(logfile, index=False)
        logfile.write("\n")
        logfile.write(model_description)
        logfile.write("\n")
        logfile.write(name_score)
        logfile.write(id_score)

    # Return average metrics across all test samples
    return np.mean(f1_list), np.mean(precision_list), np.mean(recall_list)
