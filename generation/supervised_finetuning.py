import pandas as pd
from unsloth import UnslothTrainer, UnslothTrainingArguments
from unsloth import FastLanguageModel
from datasets import load_dataset
import torch
import finetuning_test
import experiments
import os

os.environ["HF_HOME"] = "/tmp/huggingface/"
os.environ["WANDB_DISABLED"] = "true"


max_seq_length = 8192
load_in_4bit = False
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"


# Main
def train(dataset_path: str, val_dataset_len_per: float = 0.025,
          output_path: str = "outputs/", train_name="finetuned"+model_name):
    """
        Trains a LLM model on the provided dataset.

        Args:
            dataset_path: Path to JSON dataset file
            val_dataset_len_per: Percentage of dataset to use for validation
            output_path: Directory to save trained model
            train_name: Name for the training run

        Returns:
            Trained model and tokenizer
        """

    # Initialize model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    # Apply fine-tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=64, # Rank for LoRA matrices
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "embed_tokens", "lm_head"], # Layers to apply LoRA
        lora_alpha=16, # Scaling factor
        lora_dropout=0, # Dropout rate
        bias="none", # No bias modification
        use_gradient_checkpointing="unsloth", # Memory optimization
        random_state=3407, # training seed
        use_rslora=True, # Use rank-stable LoRA
        loftq_config=None, # Quantization configuration
    )

    # Load and preprocess dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Apply chat template
    def convert_to_chat_template(n):
        conv = []
        for mes in n["messages"]:
            conv.append(mes)

        con = lambda x: {"text": tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=False)}

        chat = con(conv)
        return chat
    dataset = dataset.map(convert_to_chat_template)

    # runtime print
    print(dataset['text'][0])

    # Split dataset into training and validation sets
    dataset_len = int(len(dataset) * val_dataset_len_per)
    if 0 < dataset_len < len(dataset):
        split_sets = dataset.train_test_split(test_size=dataset_len, seed=42)

        val_dataset = split_sets['train']
        train_dataset = split_sets['train']
    else:
        train_dataset = dataset
        val_dataset = dataset

    print("Training...")
    print("Dataset len: " + str(len(train_dataset)))

    # Configure training arguments
    sft_config = UnslothTrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        eval_strategy="steps" if len(train_dataset) != len(val_dataset) else "no",
        eval_steps=2500,
        per_device_eval_batch_size=2,
        warmup_steps=1000,  # 5-10% of dataset size
        num_train_epochs=3,
        #save_strategy="steps", # not needed for evaluation
        #save_steps=1000,
        learning_rate=2e-5,
        embedding_learning_rate=1e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        seed=3407,
        run_name=train_name,
        output_dir=output_path,
        optim="adamw_8bit",
        weight_decay=0.00,
        report_to=None,
        lr_scheduler_type="constant",
    )

    # Initialize trainer and start training
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=20,
        packing=False,
        args=sft_config,
    )

    trainer.train(resume_from_checkpoint=False)

    # Saving (not needed for evaluation)
    #tokenizer.save_pretrained(output_path)
    #model.save_pretrained(output_path)
    #merged_model = model.merge_and_unload()
    #merged_model.save_pretrained(output_path + "/merged/")

    return model, tokenizer


# Load model based on the name
def load_model(load_model_name):
    """
        Loads a pre-trained language model and tokenizer for inference.

        Args:
            load_model_name (str): Name of the pre-trained model to load (e.g., "unsloth/Meta-Llama-3.1-8B-Instruct").

        Returns:
            Tuple: A tuple containing:
                - model (FastLanguageModel): The loaded model in inference mode.
                - tokenizer (Tokenizer): The corresponding tokenizer for the model.
        """

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    # Set model in inference mode
    FastLanguageModel.for_inference(model)
    return model, tokenizer

# Trains a model based on the provided dataset.
def single_train(train_name="finetuned"+model_name, dataset_path="finetuning/cti_datasets/tram/tram_rag_sentence_based_train_instructs.json", val_dataset_len_per=0.01):
    """
        Trains a model on a specified dataset with default parameters for a single run.

        Args:
            train_name (str, optional): Name for the training run/output directory. Defaults to "finetuned" + model_name.
            dataset_path (str, optional): Path to the JSON training dataset. Defaults to TRAM dataset.
            val_dataset_len_per (float, optional): Percentage of dataset to use for validation. Defaults to 0.01.

        Returns:
            Tuple: A tuple containing:
                - model (FastLanguageModel): The trained model in inference mode.
                - tokenizer (Tokenizer): The corresponding tokenizer for the model.

        Notes:
            - Wraps the `train` function with predefined output paths and training parameters.
            - Converts the model to inference mode post-training.
        """

    model, tokenizer = train(output_path="finetuning/output/" + train_name, val_dataset_len_per=val_dataset_len_per,
                             dataset_path=dataset_path, train_name=train_name)

    FastLanguageModel.for_inference(model)
    return model, tokenizer


# Evaluate a model
def inference(strategies, model_name, model = None, tokenizer = None):
    """
        Evaluates a model using predefined strategies and returns performance metrics.

        Args:
            strategies (Dict): A dictionary defining the evaluation methods (see experiments.py).
            model_name (str): Identifier for the model being evaluated.
            model (FastLanguageModel): modelname for logging.
            tokenizer (Tokenizer): The tokenizer for the model.

        Returns:
            Tuple: Evaluation metrics returned by `finetuning_test.test()`:
                - f1 (float): F1 score,
                - precision (float): Precision score,
                - recall (float): Recall score
        """
    print("Start Experiment: " + model_name + " - " + strategies["Name"])

    return finetuning_test.test(strategies, model=model, tokenizer=tokenizer, model_description=model_name)



###### MAIN #####
# Main function that starts all experiments and outputs the results in structured CSV files
if __name__ == '__main__':
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct" # load from huggingface
    print("Load Model: " + model_name)

    results = []

    #Load original Meta LLama3.1-Instruction 8B
    model_base, tokenizer = load_model(model_name)
    #Table 9: Comparison results of Generative LLM methods.
    #BOSCH Prompt-based
    f1, precision, recall = inference(experiments.experimentBosch1, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.experimentBosch1["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentBosch2, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.experimentBosch2["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentBosch3, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.experimentBosch3["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentBosch4, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.experimentBosch4["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    #TRAM Prompt-based
    f1, precision, recall = inference(experiments.experimentT1, "BaseTram", model_base, tokenizer)
    results.append({'Experiment': experiments.experimentT1["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentT2, "BaseTram", model_base, tokenizer)
    results.append({'Experiment': experiments.experimentT2["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentT3, "BaseTram", model_base, tokenizer)
    results.append({'Experiment': experiments.experimentT3["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentT4, "BaseTram", model_base, tokenizer)
    results.append({'Experiment': experiments.experimentT4["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    del model_base


    #SFT
    # TRAM
    trained_model, tokenizer = single_train("mitre_sentence_tram", dataset_path="finetuning/cti_datasets/tram/tram_sentence_based_train_instructs.json")
    #trained_model, tokenizer = load_model("finetuning/output/mitre_sentence_tram")
    f1, precision, recall = inference(experiments.experimentT1, "mitre_sentence_tram", trained_model, tokenizer)
    results.append({'Experiment': experiments.experimentT1["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentT2, "mitre_sentence_tram", trained_model, tokenizer)
    results.append({'Experiment': experiments.experimentT2["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentT3, "mitre_sentence_tram", trained_model, tokenizer)
    results.append({'Experiment': experiments.experimentT3["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentT4, "mitre_sentence_tram", trained_model, tokenizer)
    results.append({'Experiment': experiments.experimentT4["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    del trained_model


    # Bosch
    model_bosch, tokenizer = single_train("bosch_sentence", dataset_path="finetuning/cti_datasets/bosch/sentence_based_train_instructs.json")
    f1, precision, recall = inference(experiments.experimentBosch1, "bosch_sentence", model_bosch, tokenizer)
    results.append({'Experiment': experiments.experimentBosch1["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentBosch2, "bosch_sentence", model_bosch, tokenizer)
    results.append({'Experiment': experiments.experimentBosch2["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentBosch3, "bosch_sentence", model_bosch, tokenizer)
    results.append({'Experiment': experiments.experimentBosch3["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentBosch4, "bosch_sentence", model_bosch, tokenizer)
    results.append({'Experiment': experiments.experimentBosch4["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    del model_bosch

    pd.DataFrame(results).to_csv("finetuning/experiments/" + "table9_methods" + ".csv", index=False)
    results = []


    # Load original Meta LLama3.1-Instruction 8B
    model_base, tokenizer = load_model(model_name)
    #Figure 4 - open-set scenario
    f1, precision, recall = inference(experiments.BoschTTPAll, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.BoschTTPAll["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.BoschTTPOpen, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.BoschTTPOpen["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.BoschTTP1, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.BoschTTP1["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.BoschTTP2, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.BoschTTP2["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.BoschTTP4, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.BoschTTP4["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    pd.DataFrame(results).to_csv("finetuning/experiments/" + "figure4_open_set" + ".csv", index=False)
    results = []

    # Cleanup
    del model_base
    torch.cuda.empty_cache()


    #Augmented Data
    # Table 10
    trained_model, tokenizer = single_train("tram_sentence_augmented", dataset_path="finetuning/cti_datasets/tram/sentence_based_augmented_train_instructs.json")
    #trained_model, tokenizer = load_model("finetuning/output/tram_sentence_augmented2")
    f1, precision, recall = inference(experiments.experimentT1, "tram_sentence_augmented", trained_model, tokenizer)
    results.append({'Experiment': experiments.experimentT1["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentT2, "tram_sentence_augmented", trained_model, tokenizer)
    results.append({'Experiment': experiments.experimentT2["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentT3, "tram_sentence_augmented", trained_model, tokenizer)
    results.append({'Experiment': experiments.experimentT3["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.experimentT4, "tram_sentence_augmented", trained_model, tokenizer)
    results.append({'Experiment': experiments.experimentT4["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    pd.DataFrame(results).to_csv("finetuning/experiments/" + "table10_augmented_data" + ".csv", index=False)
    results = []
    del trained_model


    # Document-Level Tests
    # Table 11: Document-level granularity for LLM strategies in comparison with sentence-level granularity in parenthesis.
    torch.cuda.empty_cache()
    model_base, tokenizer = load_model("unsloth/Meta-Llama-3.1-8B-Instruct")
    f1, precision, recall = inference(experiments.BoschDocumentLevelFSP, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.BoschDocumentLevelFSP["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.BoschDocumentLevelRAG, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.BoschDocumentLevelRAG["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.BoschDocumentLevel, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.BoschDocumentLevel["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.TramDocumentLevel, "BaseTram", model_base, tokenizer)
    results.append({'Experiment': experiments.TramDocumentLevel["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.TramDocumentLevelRAG, "BaseTram", model_base, tokenizer)
    results.append({'Experiment': experiments.TramDocumentLevelRAG["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    f1, precision, recall = inference(experiments.TramDocumentLevelFSP, "BaseTram", model_base, tokenizer)
    results.append({'Experiment': experiments.TramDocumentLevelFSP["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})
    del model_base

    pd.DataFrame(results).to_csv("finetuning/experiments/" + "table11_document_level" + ".csv", index=False)

