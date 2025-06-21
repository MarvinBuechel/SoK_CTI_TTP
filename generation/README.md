# TTP Extraction with Meta-Llama-3.1-8B-Instruct

This repository contains code for fine-tuning and testing large language models (LLMs) like Meta-Llama-3.1-8B-Instruct for TTP (Tactic, Technique, Procedure) extraction from CTI reports. 

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Prerequisites](#prerequisites)
3. [Setup](#setup)
4. [Training](#training)
5. [Running Experiments](#running-experiments)
6. [Results & Outputs](#results--outputs)
7. [Adding/Modifying Experiments](#addingmodifying-experiments)
8. [Notes](#notes)

---

## Getting Started

To reproduce all experiments of the paper, we recommend the one-click Docker solution.

---

## One-Click Docker runtime

To reproduce all experiments of the main part of the paper we recommend the docker-compose solution.

All you need is a server with the GPU requirements, docker and a fast internet connection. 

To start enter the following in the subfolder "generative":
 ```bash
docker compose up --build -d
 ```

Two docker containers are built, (1) the Python environment with all experiments and the generative LLMs and (2) a preconfigured ollama server which downloads the model `gte-qwen2-7b-instruct:f16` directly after starting. Everything happens in the background. With `docker log` you can access the output where you can see the experiments running and all results. All results are stored in the generative/experiments folder.

This also creates “table_12_xyz.csv” files for the results in the tables.

In the environment with the experiments, the Meta LLama3.1 8B model is automatically downloaded from huggingface at the beginning and loaded into the VRAM. All experiments then start. The runtime for us was about 2 days.

Before RAG experiments start, it takes about half an hour, until then the 16GB of the RAG model should be downloaded. If this is not the case, the RAG experiment will fail. In this case you should wait until the 16GB have finished downloading and then restart everything with `docker compose up --build -d`. Already downloaded models remain.

---

## Prerequisites for manual installation
- **Python**: 3.10 
- **GPU with 48GB VRAM**: Required for GPU acceleration (default device set to `CUDA_VISIBLE_DEVICES=1` in the script)
- **RAG-Modell**: Open-AI API compatible runtime e.g. ollama with the model `gte-qwen2-7b-instruct:f16`. Change ollama URL with the environment variable: `OLLAMA_API_URL`
- **Dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
  In addition, the unsloth library with its own optimizations must be installed for faster results. See: https://docs.unsloth.ai/#quickstart
---

## Setup Manual
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd repository-folder
   ```

2. **Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Training
### Default Experiments
To run all experiments, execute:
```bash
python supervised_finetuning.py
```
- **Models**: Trained models are saved in `finetuning/output/`.
- **Results**: Performance metrics and predictions are stored in `finetuning/experiments/` (CSV format).

### Custom Training
Modify the `single_train` function parameters in the script to adjust:
- `dataset_path`: Path to your training data (e.g., `finetuning/cti_datasets/bosch/sentence_based_train_instructs.json`).
- `output_path`: Directory for saving model checkpoints.
- Hyperparameters in `train()` (e.g., `max_seq_length`, learning rate).

---


## Results & Outputs
- **Trained Models**: Located in `finetuning/output/`.
- **Performance Reports**: CSV files in `finetuning/experiments/` include:
  - Accuracy, precision, recall, F1 scores.
  - Predictions and ground truth labels.

---

## Adding/Modifying Experiments
1. **Add a new experiment**:
   - Define a new experiment function in `experiments.py`.
   - Add a call to `inference()` in the `__main__` block with the new experiment name.

2. **Remove an experiment**: Comment out the corresponding `inference()` call in `__main__`.

---

## Notes
- **Model Access**: The `Meta-Llama-3.1-8B-Instruct` model is accessed via HuggingFace. No special token is required.
- **GPU Memory**: The 8B model may require significant VRAM. We trained it on a NVIDIA L40 with 48GB VRAM. Use `load_in_4bit=True` (in `FastLanguageModel.from_pretrained`) for memory efficiency.
- **CUDA Setup**: The script defaults to `CUDA_VISIBLE_DEVICES=1`. Adjust this variable in the script if using a different GPU.
- **Cleanup**: The script includes `torch.cuda.empty_cache()` to free memory between experiments.
- **Documentation**: Review the `unsloth` library documentation for advanced configuration options (e.g., LoRA parameters).

