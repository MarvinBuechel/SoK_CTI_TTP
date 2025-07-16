import pandas as pd
import os

# set dummy url for ollama
os.environ['OLLAMA_API_URL'] = 'dummy.com'

import supervised_finetuning as sft
import experiments


if __name__ == '__main__':
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct" # load from huggingface
    print("Load Model: " + model_name)

    results = []

    #Load original Meta LLama3.1-Instruction 8B
    model_base, tokenizer = sft.load_model(model_name, quant_4bit_model=True)

    f1, precision, recall = sft.inference(experiments.experimentBosch1, "BaseBosch", model_base, tokenizer)
    results.append({'Experiment': experiments.experimentBosch1["Name"], 'F1-Score': f1, 'Precision': precision, 'Recall': recall})

    pd.DataFrame(results).to_csv("finetuning/experiments/" + "minimal_working_example" + ".csv", index=False)

